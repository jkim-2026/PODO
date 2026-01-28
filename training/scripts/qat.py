import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import Ultralytics parts
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionTrainer

# Import PyTorch Quantization
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

def get_args():
    parser = argparse.ArgumentParser(description='Native PyTorch QAT for YOLOv8')
    parser.add_argument('--epochs', type=int, default=10, help='Number of QAT epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--weights', type=str, default=None, help='Path to best.pt')
    # Default config is now in ../configs/config.yaml
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Path to config.yaml')
    return parser.parse_args()

def main():
    args = get_args()
    print("🚀 Starting Native QAT Training (Forward + Backward)...")
    
    # [PATH RESOLUTION] Determine Project Root (training/)
    # Since this script is in training/scripts/, the root is one level up.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    print(f"📂 Project Root: {project_root}")

    # [STABILITY] 1. Force Single Thread & CPU
    # This is critical to avoid 'Backend Fallback' crashes on x86 servers.
    torch.set_num_threads(1)
    device = torch.device("cpu")
    print(f"✅ Training Config: Device={device}, Threads=1 (Stability Optimized)")

    # [CONFIG] Load user config
    # Resolve config path relative to script if it's not absolute
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)
        
    if not os.path.exists(config_path):
        print(f"❌ Config not found at: {config_path}")
        return
        
    with open(config_path, 'r') as f:
        user_config = yaml.safe_load(f)

    # [DATA PATH FIX] Resolve data_path relative to Project Root
    if 'data_path' in user_config and not os.path.isabs(user_config['data_path']):
        original_data_path = user_config['data_path']
        user_config['data_path'] = os.path.abspath(os.path.join(project_root, original_data_path))
        print(f"🔄 Resolved data_path: '{original_data_path}' -> '{user_config['data_path']}'")

    # [MODEL] 2. Load Model & Extract nn.Module
    weights_path = args.weights
    if not weights_path:
        # Auto-discovery logic
        candidates = [
            f"runs/detect/{user_config.get('exp_name', '')}/weights/best.pt",
            f"runs/{user_config.get('exp_name', '')}/weights/best.pt",
            "yolov8n.pt"
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break
                
    if not weights_path:
        print("⚠️  Warning: Local 'best.pt' not found. Defaulting to 'yolov8n.pt' for verification.")
        weights_path = "yolov8n.pt"
    
    print(f"📦 Loading weights from: {weights_path}")
    yolo_wrapper = YOLO(weights_path)
    model = yolo_wrapper.model # This is the raw nn.Module
    model.to(device).float()   # Ensure FP32
    
    # [IMPORTANT] Enable gradients for QAT fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # [FUSION] 3. Fuse Conv+BN
    print("🔧 Fusing Conv+BN layers...")
    # Use wrapper's utility to fuse the internal model
    yolo_wrapper.fuse() 
    
    # [QAT] 4. Setup Quantization
    print("⚙️  Setting up QAT (backend: fbgemm, scheme: symmetric)...")
    model.train() # Required for prepare_qat
    
    # [TENSORRT COMPATIBILITY] 
    # TensorRT (especially on Jetson) requires Symmetric Quantization (zero_point=0) 
    # to utilize hardware acceleration efficienty.
    # default_qat_qconfig('fbgemm') is asymmetric for activations. 
    # We define a custom one here.
    from torch.ao.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
    from torch.ao.quantization import QConfig, FakeQuantize
    
    # Activation: Symmetric, Per-tensor
    act_observer = MovingAverageMinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric, 
        dtype=torch.quint8,
    )
    
    # Weight: Symmetric, Per-channel
    # [FIX] MovingAverageMinMaxObserver does NOT support per-channel.
    # We MUST use MovingAveragePerChannelMinMaxObserver here.
    weight_observer = MovingAveragePerChannelMinMaxObserver.with_args(
        qscheme=torch.per_channel_symmetric,
        dtype=torch.qint8
    )
    
    # QConfig for TRT (Symmetric)
    trt_qat_qconfig = QConfig(
        activation=FakeQuantize.with_args(observer=act_observer, quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_symmetric),
        weight=FakeQuantize.with_args(observer=weight_observer, quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    )
    
    # [STRATEGY] Backbone-Only Quantization (Safest)
    model.qconfig = None
    
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        print("🛡️  Applying Symmetric QAT Strategy: Backbone Only (Layers 0-9)")
        for i, layer in enumerate(model.model):
            if i < 10: 
                layer.qconfig = trt_qat_qconfig
                print(f"   - Layer {i}: {type(layer).__name__} -> INT8 (Symmetric)")
            else:
                layer.qconfig = None
                print(f"   - Layer {i}: {type(layer).__name__} -> FP32 (Skipped)")
    else:
        print("⚠️  Warning: Model structure unknown. Applying Global Symmetric QAT.")
        model.qconfig = trt_qat_qconfig
            
    prepare_qat(model, inplace=True)
    print("✅ Model prepared for QAT.")

    # [DATA] 5. Setup DataLoader (Borrow from Ultralytics)
    print("📚 Preparing Data Loader...")
    data_yaml = os.path.join(user_config['data_path'], 'data.yaml')
    
    # [FIX] Use imgsz from config (default to 640 if not set)
    qat_imgsz = user_config.get('img_size', 640)
    print(f"🖼️  Using Image Size: {qat_imgsz}")

    # Create a dummy trainer just to get the dataloader
    overrides = {'data': data_yaml, 'imgsz': qat_imgsz, 'workers': 0, 'amp': False, 'model': weights_path}
    trainer = DetectionTrainer(overrides=overrides)
    # Hack: Inject our model so trainer doesn't try to load one (though constructor already loaded one, we overwrite it or just use the loader)
    trainer.model = model 
    
    # Get loader (train mode)
    loader = trainer.get_dataloader(
        check_det_dataset(data_yaml)['train'], 
        batch_size=args.batch_size, 
        rank=-1, 
        mode='train'
    )

    # [TRAIN] 6. Native Training Loop
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Init loss and connect hypers for v8DetectionLoss
    loss_fn = v8DetectionLoss(model) 
    
    # [FIX] Wrap dict in SimpleNamespace because loss_fn uses dot access (hyp.box)
    from types import SimpleNamespace
    
    if hasattr(trainer, 'hyp'):
        hyp_dict = trainer.hyp
    else:
        hyp_dict = user_config
        
    # Ensure it's a namespace
    if isinstance(hyp_dict, dict):
        loss_fn.hyp = SimpleNamespace(**hyp_dict)
    else:
        loss_fn.hyp = hyp_dict
        
    print(f"\n▶️  Starting Fine-tuning loop for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # 6.1 Preprocess (Normalize)
            imgs = batch['img'].to(device).float() / 255.0
            
            # 6.2 Forward
            preds = model(imgs)
            
            # 6.3 Loss
            # v8DetectionLoss expects (preds, batch)
            loss, loss_items = loss_fn(preds, batch)
            
            # 6.4 Backward
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            
            # Log
            pbar.set_postfix(loss=f"{loss.sum().item():.4f}")

        # [BEST PRACTICE] Observer Freeze & BN Freeze
        # Standard QAT strategy: Freeze observers and BN stats in the last few epochs
        # to allow the model to settle with fixed quantization parameters.
        if epoch == args.epochs - 3: 
            print("\n🧊 Freezing Quantization Observers (Scale/Zero-point fixed)...")
            model.apply(torch.ao.quantization.disable_observer)
        if epoch == args.epochs - 2:
            print("🧊 Freezing Batch Normalization statistics...")
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # [EXPORT] 7. Save & Export
    print("\n💾 Saving Quantized Model...")
    
    # [LOGIC] Infer Experiment Name from Weights Path
    # If weights are in 'runs/exp_name/weights/best.pt', we want 'exp_name'.
    # This decouples the output folder from config.yaml, preventing overwrites or mismatches.
    derived_exp_name = None
    if args.weights:
        # Try to find the parent directory of 'weights' folder
        # Path: .../runs/exp_name/weights/best.pt
        weights_dir = os.path.dirname(os.path.abspath(args.weights)) # .../runs/exp_name/weights
        if os.path.basename(weights_dir) == 'weights':
            exp_dir = os.path.dirname(weights_dir) # .../runs/exp_name
            derived_exp_name = os.path.basename(exp_dir) # exp_name
            print(f"ℹ️  Derived experiment name from weights path: '{derived_exp_name}'")

    # Fallback to config if derivation failed
    target_exp_name = derived_exp_name if derived_exp_name else user_config.get('exp_name', 'qat_native')
    
    # Append '_qat' to distinguish
    # Force 'runs' to be in the Project Root/runs
    save_dir = os.path.join(project_root, "runs", f"{target_exp_name}_qat")
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to INT8 (Eval mode)
    model.eval()
    
    # [CRITICAL] Save FakeQuant Model (For TensorRT / ONNX Export with Q/DQ nodes)
    # We do NOT run `convert()` here for the checkpoint. 
    # TensorRT needs the FakeQuant nodes (Q/DQ) to perform Explicit Quantization.
    
    # [FIX] Option A: Save ONLY state_dict (Robust & Standard)
    # We do NOT save the full model object to avoid pickling errors with QAT local functions.
    # To load this: Build model -> qconfig -> prepare -> load_state_dict
    
    pt_path = os.path.join(save_dir, "best_qat.pt")
    
    ckpt = {
        'model': model.state_dict(),
        'qat': True,
        'backend': 'fbgemm',
        'epoch': -1,
        'description': 'QAT state_dict. Must load into prepared QAT model.'
    }
    torch.save(ckpt, pt_path)
    print(f"✅ Saved .pt (State Dict Only): {pt_path}")

    # Remove the CPU-only conversion check as it's confusing and not needed for TRT

    # Export ONNX (Native)
    onnx_path = os.path.join(save_dir, "best_qat.onnx")
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    try:
        print("📦 Exporting to ONNX via Native PyTorch...")
        # [FIX] To resolve 'aten::fused_moving_avg_obs_fake_quant' error:
        # 1. Ensure absolute eval mode
        # 2. Disable observers and fake_quant grad tracking
        # 3. Use training=torch.onnx.TrainingMode.EVAL to force decomposition
        model.eval()
        model.apply(torch.ao.quantization.disable_observer)
        
        torch.onnx.export(
            model, 
            dummy_input,
            onnx_path,
            opset_version=13,
            training=torch.onnx.TrainingMode.EVAL, # CRITICAL: avoid training-only fused ops
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        print(f"✅ Saved .onnx: {onnx_path}")
    except Exception as e:
        print(f"⚠️ ONNX Export warning: {e}")
        print("Note: If this still fails, try exporting on the Jetson device where the environment may differ.")

    print("\n🎉 QAT Completed Successfully!")

if __name__ == "__main__":
    main()
