#!/usr/bin/env python3
"""Convert DroneVehicle polygon XML annotations to COCO detection JSON."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "truck", "supercategory": "vehicle"},
    {"id": 3, "name": "bus", "supercategory": "vehicle"},
    {"id": 4, "name": "van", "supercategory": "vehicle"},
    {"id": 5, "name": "freight_car", "supercategory": "vehicle"},
]

CLASS_ALIASES = {
    "car": "car",
    "truck": "truck",
    "truvk": "truck",
    "bus": "bus",
    "van": "van",
    "feright car": "freight_car",
    "feright_car": "freight_car",
    "feright": "freight_car",
    "freight car": "freight_car",
    "freight_car": "freight_car",
}


def convert(image_dir: Path, label_dir: Path, output: Path) -> None:
    category_ids = {item["name"]: item["id"] for item in CATEGORIES}
    images = []
    annotations = []
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    annotation_id = 1

    xml_files = sorted(label_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(f"No XML files found in {label_dir}")

    for image_id, xml_path in enumerate(xml_files, start=1):
        image_path = image_dir / f"{xml_path.stem}.jpg"
        if not image_path.is_file():
            skipped["missing_image"] += 1
            continue

        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(float(size.findtext("width", "0"))) if size is not None else 0
        height = int(float(size.findtext("height", "0"))) if size is not None else 0
        if width <= 0 or height <= 0:
            skipped["invalid_image_size"] += 1
            continue

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for obj in root.findall("object"):
            raw_name = (obj.findtext("name") or "").strip().lower()
            name = CLASS_ALIASES.get(raw_name)
            if name is None:
                skipped[f"class:{raw_name or '<empty>'}"] += 1
                continue

            polygon = obj.find("polygon")
            bndbox = obj.find("bndbox")
            try:
                if polygon is not None:
                    xs = [float(polygon.findtext(f"x{i}")) for i in range(1, 5)]
                    ys = [float(polygon.findtext(f"y{i}")) for i in range(1, 5)]
                    segmentation = [[value for point in zip(xs, ys) for value in point]]
                elif bndbox is not None:
                    x1 = float(bndbox.findtext("xmin"))
                    y1 = float(bndbox.findtext("ymin"))
                    x2 = float(bndbox.findtext("xmax"))
                    y2 = float(bndbox.findtext("ymax"))
                    xs = [x1, x2, x2, x1]
                    ys = [y1, y1, y2, y2]
                    segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                else:
                    skipped["unsupported_geometry"] += 1
                    continue
            except (TypeError, ValueError):
                skipped["invalid_geometry"] += 1
                continue

            x_min = max(0.0, min(xs))
            y_min = max(0.0, min(ys))
            x_max = min(float(width), max(xs))
            y_max = min(float(height), max(ys))
            box_width = x_max - x_min
            box_height = y_max - y_min
            if box_width <= 0 or box_height <= 0:
                skipped["empty_box"] += 1
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[name],
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "segmentation": segmentation,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            counts[name] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "info": {"description": "DroneVehicle paired RGB-IR detection dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    output.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    print(f"output={output}")
    print(f"images={len(images)} annotations={len(annotations)}")
    print("classes=" + json.dumps(dict(sorted(counts.items())), sort_keys=True))
    print("skipped=" + json.dumps(dict(sorted(skipped.items())), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.images, args.labels, args.output)


if __name__ == "__main__":
    main()
