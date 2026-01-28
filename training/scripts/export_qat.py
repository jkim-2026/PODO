import os
import torch
import argparse
import yaml
from ultralytics import YOLO
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat

def get_args():
    parser = argparse.ArgumentParser(description='Export QAT State Dict to ONNX')
    parser.add_argument('--weights', type=str, required=True, help='Path to best_qat.pt (State Dict)')
    parser.add_argument('--base-weights', type=str, default='yolov8n.pt', help='Base model (to get architecture)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--output', type=str, default=None, help='Output ONNX path')
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Build Base Model Architecture
    print(f"🏗️  Building base model architecture from: {args.base_weights}")
    model_wrapper = YOLO(args.base_weights)
    # [CRITICAL] Fuse Conv+BN to match the QAT training structure
    model_wrapper.fuse()
    model = model_wrapper.model
    model.float().eval()
    
    # 2. Setup QAT (SAME strategy as training: Backbone-Only)
    print("⚙️  Setting up QAT Configuration (Backbone-Only)...")
    model.qconfig = None
    for i, layer in enumerate(model.model):
        if i < 10:
             layer.qconfig = get_default_qat_qconfig('fbgemm')
        else:
             layer.qconfig = None
    
    # Prepare (Insert FakeQuant nodes)
    model.train() # Required for prepare_qat
    prepare_qat(model, inplace=True)
    
    # 3. Load State Dict
    print(f"📂 Loading QAT state_dict from: {args.weights}")
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print("✅ Weights loaded successfully.")
    
    # 4. Export to ONNX
    output_path = args.output if args.output else args.weights.replace('.pt', '.onnx')
    print(f"📦 Exporting to ONNX: {output_path}")
    
    model.eval()
    # Apply disable_observer to remove fused moving average ops for ONNX compatibility
    model.apply(torch.ao.quantization.disable_observer)
    
    # [FIX] Sanitize FakeQuantize modules for ONNX compatibility
    # Complex/Fused observers often break ONNX export. We replace them with simple functional versions.
    print("🛠️  Sanitizing FakeQuantize modules for ONNX...")
    
    class ONNXFriendlyFakeQuantize(torch.nn.Module):
        def __init__(self, orig_fq):
            super().__init__()
            self.register_buffer('scale', orig_fq.scale)
            # [FIX] Cast to Int32 (ONNX requirement)
            self.register_buffer('zero_point', orig_fq.zero_point.to(torch.int32))
            self.quant_min = orig_fq.quant_min
            self.quant_max = orig_fq.quant_max
            self.axis = getattr(orig_fq, 'ch_axis', -1)
            self.is_per_channel = self.axis != -1
            
        def forward(self, x):
            if self.is_per_channel:
                return torch.fake_quantize_per_channel_affine(
                    x, self.scale, self.zero_point, self.axis, self.quant_min, self.quant_max)
            else:
                return torch.fake_quantize_per_tensor_affine(
                    x, self.scale.item(), self.zero_point.item(), self.quant_min, self.quant_max)

    def replace_fq(module):
        for name, child in module.named_children():
            if 'FakeQuantize' in type(child).__name__:
                print(f"   - Replacing {name} ({type(child).__name__}) with ONNX-friendly version")
                setattr(module, name, ONNXFriendlyFakeQuantize(child))
            else:
                replace_fq(child)
    
    replace_fq(model)
    
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz)
    
    try:
        print("🛠️  Attempting ONNX export (opset 13)...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=13,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"✨ Successfully exported QAT ONNX to: {output_path}")
    except Exception as e:
        print(f"⚠️  standard export failed: {e}")
        print("🔄 Retrying with legacy_fused_fake_quant=False (if applicable) or alternative opset...")
        try:
            # Try higher opset?
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=16, # Higher opset might have better support
                training=torch.onnx.TrainingMode.EVAL,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output']
            )
            print(f"✨ Successfully exported QAT ONNX (opset 16) to: {output_path}")
        except Exception as e2:
             print(f"❌ All ONNX Export attempts failed: {e2}")

if __name__ == "__main__":
    main()
