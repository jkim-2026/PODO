import aiosqlite
import json
from typing import List, Dict, Optional
from schemas.schemas import DetectRequest, StatsResponse, InspectionLogResponse
from config.settings import DB_PATH


async def init_db():
    """
    Initialize database and create tables if they don't exist.
    Called on server startup.
    Schema Update: 1 row per image. 'detections' column stores JSON list of defects.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        # 세션 테이블 생성
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT
            )
        """)

        # 검사 로그 테이블 생성
        await db.execute("""
            CREATE TABLE IF NOT EXISTS inspection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_id TEXT NOT NULL,
                result TEXT NOT NULL,
                detections TEXT,
                image_path TEXT,
                session_id INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # 기존 테이블에 session_id 컬럼이 없으면 추가 (마이그레이션)
        try:
            await db.execute("ALTER TABLE inspection_logs ADD COLUMN session_id INTEGER")
        except Exception:
            # 컬럼이 이미 존재하면 무시
            pass

        await db.commit()


async def add_inspection_log(data: DetectRequest, image_path: Optional[str] = None) -> List[int]:
    """
    Save inspection result as a single row.
    'detections' list is serialized to JSON string.
    Returns: List containing the single inserted row ID.
    """
    async with aiosqlite.connect(DB_PATH) as db:

        # Serialize detections list to JSON string
        # data.detections is a list of Pydantic models, so we dump them to dicts
        detections_json = json.dumps([d.model_dump() for d in data.detections])

        result_status = "defect" if len(data.detections) > 0 else "normal"

        cursor = await db.execute(
            """
            INSERT INTO inspection_logs (timestamp, image_id, result, detections, image_path, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                data.timestamp,
                data.image_id,
                result_status,
                detections_json,
                image_path,
                data.session_id
            )
        )
        await db.commit()

        return [cursor.lastrowid]


async def get_stats(session_id: Optional[int] = None) -> StatsResponse:
    """
    Calculate statistics from the logs.
    Now requires parsing 'detections' column to count total defects.
    session_id가 제공되면 해당 세션의 통계만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 세션 필터 조건
        session_filter = ""
        session_params = ()
        if session_id is not None:
            session_filter = " WHERE session_id = ?"
            session_params = (session_id,)

        # 1. Total Inspections (Total Rows)
        cursor = await db.execute(
            f"SELECT COUNT(*) as cnt FROM inspection_logs{session_filter}",
            session_params
        )
        row = await cursor.fetchone()
        total_inspections = row["cnt"]

        if total_inspections == 0:
            return StatsResponse(
                total_inspections=0,
                normal_count=0,
                defect_items=0,
                total_defects=0,
                defect_rate=0.0,
                avg_defects_per_item=0.0,
                avg_fps=0.0,
                last_defect=None
            )

        # 2. Defect Items (Rows where result='defect')
        defect_filter = " WHERE result='defect'" if not session_filter else " WHERE result='defect' AND session_id = ?"
        defect_params = () if not session_filter else (session_id,)
        cursor = await db.execute(
            f"SELECT COUNT(*) as cnt FROM inspection_logs{defect_filter}",
            defect_params
        )
        row = await cursor.fetchone()
        defect_items = row["cnt"]

        # 3. Total Defects (Sum of length of detections list in each defect row)
        # SQLite JSON extension might not be enabled, so we fetch and sum in Python.
        cursor = await db.execute(
            f"SELECT detections FROM inspection_logs{defect_filter}",
            defect_params
        )
        rows = await cursor.fetchall()

        total_defects = 0
        for r in rows:
            try:
                dets = json.loads(r["detections"])
                total_defects += len(dets)
            except:
                pass

        normal_count = total_inspections - defect_items
        defect_rate = (defect_items / total_inspections) * 100
        avg_defects_per_item = total_defects / defect_items if defect_items > 0 else 0.0
        avg_fps = 0.0

        # 4. Last Defect Log
        cursor = await db.execute(
            f"SELECT * FROM inspection_logs{defect_filter} ORDER BY id DESC LIMIT 1",
            defect_params
        )
        row = await cursor.fetchone()

        last_defect_log = None
        if row:
            # Map DB row to InspectionLogResponse
            last_defect_log = InspectionLogResponse(
                id=row["id"],
                timestamp=row["timestamp"],
                image_id=row["image_id"],
                result=row["result"],
                detections=json.loads(row["detections"]) if row["detections"] else [],
                image_path=row["image_path"],
                session_id=row["session_id"]
            )

        return StatsResponse(
            total_inspections=total_inspections,
            normal_count=normal_count,
            defect_items=defect_items,
            total_defects=total_defects,
            defect_rate=round(defect_rate, 2),
            avg_defects_per_item=round(avg_defects_per_item, 2),
            avg_fps=avg_fps,
            last_defect=last_defect_log
        )


async def get_recent_logs(limit: int = 10, session_id: Optional[int] = None) -> List[Dict]:
    """
    Returns the N most recent logs.
    session_id가 제공되면 해당 세션의 로그만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if session_id is not None:
            cursor = await db.execute(
                "SELECT * FROM inspection_logs WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM inspection_logs ORDER BY id DESC LIMIT ?",
                (limit,)
            )
        rows = await cursor.fetchall()

        result = []
        for row in rows:
            record = dict(row)
            if record["detections"]:
                record["detections"] = json.loads(record["detections"])
            else:
                record["detections"] = []
            result.append(record)

        return result


async def get_defect_logs(session_id: Optional[int] = None) -> List[Dict]:
    """
    Returns all logs that are defects.
    session_id가 제공되면 해당 세션의 결함 로그만 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        if session_id is not None:
            cursor = await db.execute(
                "SELECT * FROM inspection_logs WHERE result = 'defect' AND session_id = ? ORDER BY id DESC",
                (session_id,)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM inspection_logs WHERE result = 'defect' ORDER BY id DESC"
            )
        rows = await cursor.fetchall()

        result = []
        for row in rows:
            record = dict(row)
            if record["detections"]:
                record["detections"] = json.loads(record["detections"])
            else:
                record["detections"] = []
            result.append(record)

        return result


# ===== 세션 관리 함수 =====

async def create_session() -> Dict:
    """
    새 세션을 생성하고 ID와 시작 시간을 반환.
    """
    from datetime import datetime
    started_at = datetime.now().isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO sessions (started_at) VALUES (?)",
            (started_at,)
        )
        await db.commit()
        session_id = cursor.lastrowid

    return {"id": session_id, "started_at": started_at, "ended_at": None}


async def end_session(session_id: int) -> Optional[Dict]:
    """
    세션을 종료하고 ended_at을 설정.
    """
    from datetime import datetime
    ended_at = datetime.now().isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 세션 존재 여부 확인
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        # ended_at 업데이트
        await db.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (ended_at, session_id)
        )
        await db.commit()

        return {
            "id": session_id,
            "started_at": row["started_at"],
            "ended_at": ended_at
        }


async def get_sessions() -> List[Dict]:
    """
    모든 세션 목록을 반환 (최신순).
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions ORDER BY id DESC"
        )
        rows = await cursor.fetchall()

        return [dict(row) for row in rows]


async def get_session(session_id: int) -> Optional[Dict]:
    """
    특정 세션 정보를 반환.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()

        if row:
            return dict(row)
        return None
