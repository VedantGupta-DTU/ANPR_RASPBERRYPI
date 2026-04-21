#!/usr/bin/env python3
"""
ANPR Database Query Tool
========================
CLI utility to query and export plate detection results from the SQLite database.
Designed for use on Raspberry Pi 5.

Usage:
    python db_query.py --all                          # View all plates
    python db_query.py --plate "DL 9"                 # Search by plate
    python db_query.py --since "2026-04-09"           # Filter by date
    python db_query.py --stats                        # Show statistics
    python db_query.py --recent 10                    # Last 10 detections
    python db_query.py --export results_export.csv    # Export to CSV
"""

import argparse
import json
import sys

import db as anpr_db


def print_table(rows, max_width=120):
    """Pretty-print a list of dicts as an aligned table."""
    if not rows:
        print("  (no records)")
        return

    # Select useful columns for display
    display_cols = [
        "id", "timestamp", "plate", "det_conf", "ocr_conf",
        "source", "source_type", "num_reads",
    ]
    # Filter to columns that actually exist in the data
    cols = [c for c in display_cols if c in rows[0]]

    # Calculate column widths
    widths = {}
    for col in cols:
        widths[col] = max(len(col), max(len(str(row.get(col, ""))) for row in rows))
        widths[col] = min(widths[col], 30)  # cap

    # Header
    header = " | ".join(col.ljust(widths[col])[:widths[col]] for col in cols)
    separator = "-+-".join("-" * widths[col] for col in cols)
    print(f"  {header}")
    print(f"  {separator}")

    # Rows
    for row in rows:
        line = " | ".join(
            str(row.get(col, "")).ljust(widths[col])[:widths[col]]
            for col in cols
        )
        print(f"  {line}")


def cmd_list(args):
    """List/search plate detections."""
    results = anpr_db.query_plates(
        plate_filter=args.plate,
        start_time=args.since,
        end_time=args.until,
        source=args.source,
        source_type=args.type,
        min_det_conf=args.min_det,
        min_ocr_conf=args.min_ocr,
        limit=args.limit,
        offset=args.offset,
    )
    print(f"\n  Found {len(results)} record(s)\n")
    print_table(results)
    print()


def cmd_stats(args):
    """Show database statistics."""
    stats = anpr_db.get_stats()
    print(f"\n{'='*50}")
    print(f"  ANPR Database Statistics")
    print(f"{'='*50}")
    print(f"  Total detections : {stats['total_detections']}")
    print(f"  Unique plates    : {stats['unique_plates']}")
    print(f"  Valid detections : {stats['valid_detections']}")
    print(f"  Sources          : {', '.join(stats['sources']) if stats['sources'] else 'none'}")
    print(f"  Source types     : {', '.join(stats['source_types']) if stats['source_types'] else 'none'}")
    print(f"  Time range       : {stats['earliest'] or 'N/A'} → {stats['latest'] or 'N/A'}")
    print(f"  Avg det. conf    : {stats['avg_det_conf'] or 'N/A'}")
    print(f"  Avg OCR conf     : {stats['avg_ocr_conf'] or 'N/A'}")

    if stats.get("top_plates"):
        print(f"\n  Top detected plates:")
        for p in stats["top_plates"]:
            print(f"    {p['plate']:20s} — {p['count']} detection(s)")

    print(f"{'='*50}\n")


def cmd_recent(args):
    """Show most recent detections."""
    results = anpr_db.get_recent_plates(limit=args.recent)
    print(f"\n  Last {len(results)} detection(s)\n")
    print_table(results)
    print()


def cmd_export(args):
    """Export to CSV."""
    count = anpr_db.export_csv(
        args.export,
        plate_filter=args.plate,
        start_time=args.since,
        end_time=args.until,
        source=args.source,
        source_type=args.type,
    )
    print(f"\n  Exported {count} record(s) → {args.export}\n")


def cmd_json(args):
    """Output results as JSON."""
    results = anpr_db.query_plates(
        plate_filter=args.plate,
        start_time=args.since,
        end_time=args.until,
        limit=args.limit,
    )
    print(json.dumps(results, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="ANPR Database Query Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python db_query.py --all
  python db_query.py --plate "DL 9 CBC"
  python db_query.py --since "2026-04-09" --until "2026-04-10"
  python db_query.py --stats
  python db_query.py --recent 10
  python db_query.py --export output.csv
  python db_query.py --plate "HR" --json
        """,
    )

    # Actions
    parser.add_argument("--all", action="store_true", help="List all detections")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--recent", type=int, metavar="N", help="Show N most recent detections")
    parser.add_argument("--export", metavar="FILE", help="Export results to CSV file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Filters
    parser.add_argument("--plate", metavar="TEXT", help="Search by plate (substring)")
    parser.add_argument("--since", metavar="TIME", help="Start time (ISO 8601)")
    parser.add_argument("--until", metavar="TIME", help="End time (ISO 8601)")
    parser.add_argument("--source", help="Filter by source")
    parser.add_argument("--type", choices=["live", "video", "image"], help="Filter by source type")
    parser.add_argument("--min-det", type=float, dest="min_det", help="Min detection confidence")
    parser.add_argument("--min-ocr", type=float, dest="min_ocr", help="Min OCR confidence")

    # Pagination
    parser.add_argument("--limit", type=int, default=100, help="Max results (default: 100)")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset")

    args = parser.parse_args()

    # Initialize DB
    anpr_db.init_db()

    # Dispatch
    if args.stats:
        cmd_stats(args)
    elif args.recent:
        cmd_recent(args)
    elif args.export:
        cmd_export(args)
    elif args.json:
        cmd_json(args)
    elif args.all or args.plate or args.since or args.source or args.type:
        cmd_list(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
