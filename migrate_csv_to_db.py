#!/usr/bin/env python3
"""
Migrate CSV to Database
=======================
One-time migration script to import existing result.csv data
into the new SQLite database. Safe to run multiple times
(uses INSERT, so duplicates will be added — run once only).

Usage:
    python migrate_csv_to_db.py
    python migrate_csv_to_db.py --csv result.csv
    python migrate_csv_to_db.py --csv live_app_detections.csv --source live_app
"""

import argparse
import csv
import os
import sys

import db as anpr_db


def migrate_result_csv(csv_path: str, source: str = "live_camera",
                       source_type: str = "live") -> int:
    """
    Import a result.csv file (format: timestamp, plate, det_conf, ocr_conf)
    into the SQLite database.
    
    Returns:
        Number of rows imported.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return 0

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("plate"):
                continue
            rows.append({
                "timestamp": row.get("timestamp", ""),
                "plate": row["plate"],
                "det_conf": float(row.get("det_conf", 0)),
                "ocr_conf": float(row.get("ocr_conf", 0)),
                "source": source,
                "source_type": source_type,
                "is_valid": True,
            })

    if not rows:
        print(f"[INFO] No data rows found in {csv_path}")
        return 0

    count = anpr_db.insert_detections_batch(rows, source_type=source_type)
    return count


def migrate_video_csv(csv_path: str) -> int:
    """
    Import a video pipeline CSV (format with video, plate, frame_index, etc.)
    into the SQLite database.
    
    Returns:
        Number of rows imported.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return 0

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("plate"):
                continue
            rows.append({
                "plate": row["plate"],
                "det_conf": float(row.get("det_conf", 0)),
                "ocr_conf": float(row.get("ocr_conf", 0)),
                "source": row.get("video", row.get("source", "unknown")),
                "source_type": "video",
                "frame_index": int(row.get("frame_index", 0)) if row.get("frame_index") else None,
                "time_sec": float(row.get("time_sec", 0)) if row.get("time_sec") else None,
                "num_reads": int(row.get("num_reads", 0)) if row.get("num_reads") else None,
                "bbox_x1": int(row.get("bbox_x1", 0)) if row.get("bbox_x1") else None,
                "bbox_y1": int(row.get("bbox_y1", 0)) if row.get("bbox_y1") else None,
                "bbox_x2": int(row.get("bbox_x2", 0)) if row.get("bbox_x2") else None,
                "bbox_y2": int(row.get("bbox_y2", 0)) if row.get("bbox_y2") else None,
                "ocr_engine": row.get("ocr_engine"),
                "is_valid": True,
            })

    if not rows:
        print(f"[INFO] No data rows found in {csv_path}")
        return 0

    count = anpr_db.insert_detections_batch(rows, source_type="video")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Migrate CSV data to ANPR SQLite database",
    )
    parser.add_argument("--csv", default="result.csv",
                        help="Path to CSV file to import (default: result.csv)")
    parser.add_argument("--source", default="live_camera",
                        help="Source label for the imported data (default: live_camera)")
    parser.add_argument("--type", default="live", choices=["live", "video", "image"],
                        dest="source_type",
                        help="Source type (default: live)")
    parser.add_argument("--video-format", action="store_true",
                        help="Use video CSV format (with frame_index, bbox, etc.)")

    args = parser.parse_args()

    # Initialize database
    anpr_db.init_db()

    csv_path = args.csv
    print(f"\n{'='*50}")
    print(f"  CSV → Database Migration")
    print(f"{'='*50}")
    print(f"  File: {csv_path}")

    if args.video_format:
        count = migrate_video_csv(csv_path)
    else:
        count = migrate_result_csv(csv_path, source=args.source,
                                    source_type=args.source_type)

    print(f"  Imported: {count} record(s)")

    # Show quick stats
    stats = anpr_db.get_stats()
    print(f"\n  Database now contains:")
    print(f"    Total detections : {stats['total_detections']}")
    print(f"    Unique plates    : {stats['unique_plates']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
