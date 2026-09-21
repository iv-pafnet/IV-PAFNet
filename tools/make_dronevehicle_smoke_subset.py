"""Build a tiny paired DroneVehicle subset for an end-to-end training smoke test.

The generated subset preserves the original image IDs, annotations, and category
metadata. It is an engineering fixture only and must not be used to report paper
metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SPLITS = {
    "train": ("trainimg", "trainimgr", "instances_train.json"),
    "val": ("valimg", "valimgr", "instances_val.json"),
}


def select_images(coco: dict, count: int) -> list[dict]:
    annotated_ids = {item["image_id"] for item in coco["annotations"]}
    candidates = [item for item in coco["images"] if item["id"] in annotated_ids]
    candidates.sort(key=lambda item: (str(item["file_name"]), int(item["id"])))
    if len(candidates) < count:
        raise ValueError(f"requested {count} annotated images, found {len(candidates)}")
    return candidates[:count]


def build_split(source: Path, destination: Path, split: str, count: int) -> None:
    visible_name, infrared_name, annotation_name = SPLITS[split]
    annotation_path = source / "coco_annotations" / annotation_name
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = select_images(coco, count)
    image_ids = {item["id"] for item in images}
    annotations = [
        item for item in coco["annotations"] if item["image_id"] in image_ids
    ]

    visible_output = destination / split / visible_name
    infrared_output = destination / split / infrared_name
    visible_output.mkdir(parents=True, exist_ok=True)
    infrared_output.mkdir(parents=True, exist_ok=True)

    for item in images:
        filename = item["file_name"]
        visible_source = source / split / visible_name / filename
        infrared_source = source / split / infrared_name / filename
        if not visible_source.is_file() or not infrared_source.is_file():
            raise FileNotFoundError(f"missing paired files for {filename}")
        shutil.copy2(visible_source, visible_output / filename)
        shutil.copy2(infrared_source, infrared_output / filename)

    filtered = {
        key: value
        for key, value in coco.items()
        if key not in {"images", "annotations"}
    }
    filtered["images"] = images
    filtered["annotations"] = annotations

    annotation_output = destination / "coco_annotations" / annotation_name
    annotation_output.parent.mkdir(parents=True, exist_ok=True)
    annotation_output.write_text(
        json.dumps(filtered, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"{split}: {len(images)} paired images, {len(annotations)} annotations -> "
        f"{annotation_output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2)
    parser.add_argument("--val-count", type=int, default=2)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output}. "
            "Choose a new directory or remove the previous generated fixture."
        )

    build_split(args.source, args.output, "train", args.train_count)
    build_split(args.source, args.output, "val", args.val_count)


if __name__ == "__main__":
    main()
