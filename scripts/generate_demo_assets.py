"""Generate deterministic synthetic wafer maps and SPC data for the public demo corpus."""

from __future__ import annotations

import csv
import hashlib
import math
import struct
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "data" / "assets"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_rgb_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    rows = b"".join(
        b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3])
        for row in range(height)
    )
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(rows, level=9))
    payload += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def generate_wafer_map(path: Path, *, edge_ring: bool) -> None:
    width = height = 320
    pixels = bytearray([244, 247, 248] * width * height)
    center_x = center_y = 160
    radius = 138

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    for y in range(height):
        for x in range(width):
            distance = math.hypot(x - center_x, y - center_y)
            if distance <= radius:
                set_pixel(x, y, (220, 229, 233))
            if radius - 2 <= distance <= radius + 1:
                set_pixel(x, y, (46, 67, 78))

    die_size = 11
    step = 14
    for die_y in range(center_y - 126, center_y + 127, step):
        for die_x in range(center_x - 126, center_x + 127, step):
            distance = math.hypot(die_x - center_x, die_y - center_y)
            if distance > radius - die_size:
                continue
            ring_defect = edge_ring and distance >= radius * 0.72
            color = (198, 65, 58) if ring_defect else (72, 143, 120)
            for y in range(die_y - die_size // 2, die_y + die_size // 2):
                for x in range(die_x - die_size // 2, die_x + die_size // 2):
                    set_pixel(x, y, color)

    for y in range(292, 303):
        for x in range(151, 170):
            if abs(x - 160) + (302 - y) < 12:
                set_pixel(x, y, (244, 247, 248))
    write_rgb_png(path, width, height, pixels)


def generate_spc_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 2, 8, 0, 0, tzinfo=UTC)
    fieldnames = [
        "timestamp",
        "lot_id",
        "wafer_id",
        "tool_id",
        "chamber",
        "recipe_id",
        "recipe_version",
        "metric_name",
        "value",
        "center_line",
        "ucl",
        "lcl",
        "is_ooc",
        "anomaly_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(120):
            value = 41.5 + 0.18 * math.sin(index / 6) + 0.06 * math.cos(index / 3)
            anomaly = 72 <= index <= 76
            if anomaly:
                value += 1.05 + 0.08 * (index - 72)
            writer.writerow(
                {
                    "timestamp": (start + timedelta(minutes=15 * index)).isoformat(),
                    "lot_id": f"LOT-SYN-{index // 25 + 1:03d}",
                    "wafer_id": f"W{index % 25 + 1:02d}",
                    "tool_id": "ETCH-03",
                    "chamber": "B",
                    "recipe_id": "ETCH-ALPHA",
                    "recipe_version": "V2.3",
                    "metric_name": "chamber_pressure_mTorr",
                    "value": f"{value:.4f}",
                    "center_line": "41.5000",
                    "ucl": "42.2000",
                    "lcl": "40.8000",
                    "is_ooc": str(anomaly).lower(),
                    "anomaly_type": "pressure_upward_shift" if anomaly else "normal",
                }
            )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    edge_map = ASSET_ROOT / "wafer_maps" / "etch03_chamber_b_edge_ring.png"
    normal_map = ASSET_ROOT / "wafer_maps" / "etch03_chamber_b_normal.png"
    spc_data = ASSET_ROOT / "spc" / "etch03_chamber_b_pressure.csv"
    generate_wafer_map(edge_map, edge_ring=True)
    generate_wafer_map(normal_map, edge_ring=False)
    generate_spc_csv(spc_data)
    for path in (edge_map, normal_map, spc_data):
        print(f"{path.relative_to(ROOT).as_posix()} {sha256(path)}")


if __name__ == "__main__":
    main()
