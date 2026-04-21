"""
ANPR Database Module — MongoDB Atlas
=====================================
Cloud-based storage for plate detection results using MongoDB Atlas.
Designed for Raspberry Pi 5 edge deployment with cloud persistence.

Usage:
    from db import init_db, insert_detection, query_plates, get_stats

    init_db()  # call once at startup
    insert_detection(plate="DL 9 CBC 2776", det_conf=0.89, ...)
"""

import datetime
import csv
from typing import Optional, List, Dict, Any

from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.errors import ConnectionFailure

import config

# ---------------------------------------------------------------------------
# MongoDB connection (singleton)
# ---------------------------------------------------------------------------

_client: Optional[MongoClient] = None
_db = None
_collection = None


def _get_collection():
    """Get the MongoDB collection, initializing if needed."""
    global _client, _db, _collection
    if _collection is None:
        init_db()
    return _collection


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db(mongo_uri: Optional[str] = None):
    """
    Initialize the MongoDB connection and create indexes.
    Call this once at application startup.

    Args:
        mongo_uri: Override the MongoDB URI (uses config.MONGO_URI by default).
    """
    global _client, _db, _collection

    uri = mongo_uri or getattr(config, "MONGO_URI", "")
    db_name = getattr(config, "MONGO_DB_NAME", "anpr_system")
    coll_name = getattr(config, "MONGO_COLLECTION", "plate_detections")

    if not uri:
        raise ValueError("MONGO_URI is not configured in config.py")

    _client = MongoClient(uri, serverSelectionTimeoutMS=5000)

    # Test connection
    try:
        _client.admin.command("ping")
        print(f"[DB] Connected to MongoDB Atlas successfully!")
    except ConnectionFailure as e:
        print(f"[DB] ✗ MongoDB connection failed: {e}")
        raise

    _db = _client[db_name]
    _collection = _db[coll_name]

    # Create indexes for fast queries
    _collection.create_index([("plate", ASCENDING)])
    _collection.create_index([("timestamp", DESCENDING)])
    _collection.create_index([("source", ASCENDING)])
    _collection.create_index([("source_type", ASCENDING)])

    print(f"[DB] Database: {db_name} | Collection: {coll_name}")
    print(f"[DB] Indexes created/verified.")


# ---------------------------------------------------------------------------
# Insert operations
# ---------------------------------------------------------------------------

def insert_detection(
    plate: str,
    det_conf: float = 0.0,
    ocr_conf: float = 0.0,
    source: str = "unknown",
    source_type: str = "live",
    frame_index: Optional[int] = None,
    time_sec: Optional[float] = None,
    num_reads: Optional[int] = None,
    is_valid: bool = True,
    bbox: Optional[List[int]] = None,
    ocr_engine: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Insert a single plate detection into MongoDB.

    Returns:
        The inserted document's _id as a string.
    """
    if not getattr(config, "DB_ENABLED", True):
        return ""

    if timestamp is None:
        timestamp = datetime.datetime.now().isoformat()

    doc = {
        "timestamp": timestamp,
        "plate": plate,
        "det_conf": det_conf,
        "ocr_conf": ocr_conf,
        "source": source,
        "source_type": source_type,
        "frame_index": frame_index,
        "time_sec": time_sec,
        "num_reads": num_reads,
        "is_valid": is_valid,
        "bbox": {
            "x1": bbox[0] if bbox and len(bbox) >= 1 else None,
            "y1": bbox[1] if bbox and len(bbox) >= 2 else None,
            "x2": bbox[2] if bbox and len(bbox) >= 3 else None,
            "y2": bbox[3] if bbox and len(bbox) >= 4 else None,
        } if bbox else None,
        "ocr_engine": ocr_engine,
        "created_at": datetime.datetime.utcnow(),
    }

    coll = _get_collection()
    result = coll.insert_one(doc)
    return str(result.inserted_id)


def insert_detections_batch(rows: List[Dict[str, Any]],
                            source_type: str = "video") -> int:
    """
    Bulk-insert multiple detection records into MongoDB.

    Returns:
        Number of documents inserted.
    """
    if not getattr(config, "DB_ENABLED", True):
        return 0

    docs = []
    for row in rows:
        plate = row.get("plate", "")
        if not plate:
            continue

        # Handle bbox as list or individual fields
        bbox_obj = None
        bbox = row.get("bbox")
        if bbox and len(bbox) >= 4:
            bbox_obj = {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
        elif any(row.get(k) is not None for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
            bbox_obj = {
                "x1": row.get("bbox_x1"),
                "y1": row.get("bbox_y1"),
                "x2": row.get("bbox_x2"),
                "y2": row.get("bbox_y2"),
            }

        docs.append({
            "timestamp": row.get("timestamp", datetime.datetime.now().isoformat()),
            "plate": plate,
            "det_conf": row.get("det_conf", 0.0),
            "ocr_conf": row.get("ocr_conf", 0.0),
            "source": row.get("source", row.get("video", "unknown")),
            "source_type": row.get("source_type", source_type),
            "frame_index": row.get("frame_index"),
            "time_sec": row.get("time_sec"),
            "num_reads": row.get("num_reads"),
            "is_valid": row.get("is_valid", True),
            "bbox": bbox_obj,
            "ocr_engine": row.get("ocr_engine"),
            "created_at": datetime.datetime.utcnow(),
        })

    if not docs:
        return 0

    coll = _get_collection()
    result = coll.insert_many(docs)
    return len(result.inserted_ids)


# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------

def query_plates(
    plate_filter: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    source: Optional[str] = None,
    source_type: Optional[str] = None,
    min_det_conf: Optional[float] = None,
    min_ocr_conf: Optional[float] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query plate detections with flexible filters.

    Returns:
        List of detection dicts.
    """
    coll = _get_collection()
    query = {}

    if plate_filter:
        query["plate"] = {"$regex": plate_filter, "$options": "i"}
    if start_time:
        query.setdefault("timestamp", {})["$gte"] = start_time
    if end_time:
        query.setdefault("timestamp", {})["$lte"] = end_time
    if source:
        query["source"] = source
    if source_type:
        query["source_type"] = source_type
    if min_det_conf is not None:
        query["det_conf"] = {"$gte": min_det_conf}
    if min_ocr_conf is not None:
        query["ocr_conf"] = {"$gte": min_ocr_conf}

    cursor = coll.find(query).sort("timestamp", DESCENDING).skip(offset).limit(limit)

    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        # Flatten bbox for compatibility
        if doc.get("bbox") and isinstance(doc["bbox"], dict):
            bbox = doc.pop("bbox")
            doc["bbox_x1"] = bbox.get("x1")
            doc["bbox_y1"] = bbox.get("y1")
            doc["bbox_x2"] = bbox.get("x2")
            doc["bbox_y2"] = bbox.get("y2")
        # Convert datetime to string
        if doc.get("created_at"):
            doc["created_at"] = doc["created_at"].isoformat() if hasattr(doc["created_at"], "isoformat") else str(doc["created_at"])
        results.append(doc)

    return results


def get_stats() -> Dict[str, Any]:
    """
    Get summary statistics from the database.

    Returns:
        Dict with total_detections, unique_plates, sources, time_range, etc.
    """
    coll = _get_collection()

    total = coll.count_documents({})
    valid_count = coll.count_documents({"is_valid": True})

    unique_plates = len(coll.distinct("plate"))
    sources = coll.distinct("source")
    source_types = coll.distinct("source_type")

    # Time range
    earliest_doc = coll.find_one({}, sort=[("timestamp", ASCENDING)])
    latest_doc = coll.find_one({}, sort=[("timestamp", DESCENDING)])

    # Average confidences
    avg_pipeline = [
        {"$group": {
            "_id": None,
            "avg_det_conf": {"$avg": "$det_conf"},
            "avg_ocr_conf": {"$avg": "$ocr_conf"},
        }}
    ]
    avg_result = list(coll.aggregate(avg_pipeline))
    avg_det = avg_result[0]["avg_det_conf"] if avg_result else None
    avg_ocr = avg_result[0]["avg_ocr_conf"] if avg_result else None

    # Top 10 most frequently detected plates
    top_pipeline = [
        {"$group": {"_id": "$plate", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]
    top_plates = [
        {"plate": doc["_id"], "count": doc["count"]}
        for doc in coll.aggregate(top_pipeline)
    ]

    return {
        "total_detections": total,
        "unique_plates": unique_plates,
        "valid_detections": valid_count,
        "sources": sources,
        "source_types": source_types,
        "earliest": earliest_doc.get("timestamp") if earliest_doc else None,
        "latest": latest_doc.get("timestamp") if latest_doc else None,
        "avg_det_conf": round(avg_det, 4) if avg_det else None,
        "avg_ocr_conf": round(avg_ocr, 4) if avg_ocr else None,
        "top_plates": top_plates,
    }


def get_recent_plates(limit: int = 20) -> List[Dict[str, Any]]:
    """Get the most recent plate detections."""
    return query_plates(limit=limit)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(
    output_path: str,
    **filters,
) -> int:
    """
    Export filtered plate detections to a CSV file.

    Returns:
        Number of rows exported.
    """
    if "limit" not in filters:
        filters["limit"] = 999999

    rows = query_plates(**filters)
    if not rows:
        print(f"[DB] No records to export.")
        return 0

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DB] Exported {len(rows)} records → {output_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def close_db():
    """Close the MongoDB connection."""
    global _client, _db, _collection
    if _client:
        _client.close()
        _client = None
        _db = None
        _collection = None
