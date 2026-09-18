#!/usr/bin/env python3
"""
Embed results/report.json into index.html

Usage:
  python embed_report.py results/report.json
  python embed_report.py results/report.json --html index.html

Looks for markers in index.html:
  <!-- REPORT_JSON_START -->
  ... replaced ...
  <!-- REPORT_JSON_END -->
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


START = "<!-- REPORT_JSON_START -->"
END = "<!-- REPORT_JSON_END -->"


def embed(report_path: Path, html_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Compact but valid JSON inside the script tag
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":"))

    html = html_path.read_text(encoding="utf-8")
    if START not in html or END not in html:
        raise SystemExit(
            f"Markers not found in {html_path}. "
            f"Need {START} … {END}"
        )

    before, rest = html.split(START, 1)
    _, after = rest.split(END, 1)
    block = (
        f"{START}\n"
        f'<script id="report-data" type="application/json">{payload}</script>\n'
        f"{END}"
    )
    html_path.write_text(before + block + after, encoding="utf-8")
    print(f"Embedded {report_path} -> {html_path}")
    print(f"  runs: {len(report.get('runs', []))}")
    print(f"  open {html_path.resolve()} in a browser")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path, help="Path to report.json from Colab")
    p.add_argument("--html", type=Path, default=Path("index.html"))
    args = p.parse_args()
    if not args.report.exists():
        raise SystemExit(f"Missing report: {args.report}")
    if not args.html.exists():
        raise SystemExit(f"Missing html: {args.html}")
    # Validate schema lightly
    data = json.loads(args.report.read_text(encoding="utf-8"))
    for key in ("schema_version", "meta", "fertility", "runs", "findings"):
        if key not in data:
            raise SystemExit(f"report.json missing key: {key}")
    embed(args.report, args.html)


if __name__ == "__main__":
    main()
