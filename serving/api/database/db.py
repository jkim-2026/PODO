from typing import List, Dict, Optional
from schemas.schemas import DetectRequest, DefectInfo, StatsResponse

# In-memory storage (Simulating a database)
# logs list stores full dictionaries of inspection results
_inspection_logs: List[Dict] = []
_id_counter = 0

async def add_inspection_log(data: DetectRequest, image_path: Optional[str] = None) -> int:
    """
    Saves the inspection result to the in-memory list.
    Returns the new record ID.
    """
    global _id_counter
    _id_counter += 1
    
    record = data.model_dump()
    record["id"] = _id_counter
    record["image_path"] = image_path  # path or None
    
    # We don't need to store the raw base64 image string in memory/DB if we have the file
    if "image" in record:
        del record["image"]
        
    _inspection_logs.append(record)
    return _id_counter

async def get_stats() -> StatsResponse:
    """
    Calculates statistics from the in-memory logs.
    """
    total_count = len(_inspection_logs)
    if total_count == 0:
        return StatsResponse(
            total_count=0,
            normal_count=0,
            defect_count=0,
            defect_rate=0.0,
            avg_fps=0.0,
            last_defect=None
        )
        
    defect_list = [log for log in _inspection_logs if log["result"] == "defect"]
    defect_count = len(defect_list)
    normal_count = total_count - defect_count
    defect_rate = (defect_count / total_count) * 100
    
    # Calculate Average FPS (Placeholder logic)
    # In a real DB, we might timestamp diffs. 
    # Here we might need a stored value or compute it if we had proper frame intervals.
    # For now, we return a mock or stored value if we tracked it.
    # Since the request doesn't send FPS, we might assume 30.0 or calculate based on recent timestamps.
    # Let's return 0.0 for now or a dummy value as we don't have 'fps' in input.
    # TODO: Discuss with Edge team to send 'fps' key or calculate interval here.
    avg_fps = 0.0 
    
    last_defect_info = None
    if defect_list:
        latest = defect_list[-1]
        last_defect_info = DefectInfo(
            timestamp=latest["timestamp"],
            image_id=latest["image_id"],
            result=latest["result"],
            confidence=latest["confidence"],
            defect_type=latest.get("defect_type"),
            bbox=latest.get("bbox"),
            image_path=latest.get("image_path")
        )

    return StatsResponse(
        total_count=total_count,
        normal_count=normal_count,
        defect_count=defect_count,
        defect_rate=round(defect_rate, 2),
        avg_fps=avg_fps, 
        last_defect=last_defect_info
    )

async def get_recent_logs(limit: int = 10) -> List[Dict]:
    """
    Returns the N most recent logs.
    """
    return _inspection_logs[-limit:]

async def get_defect_logs() -> List[Dict]:
    """
    Returns all logs that are defects.
    """
    return [log for log in _inspection_logs if log["result"] == "defect"]
