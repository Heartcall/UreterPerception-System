"""Score the frozen G0 front end on independent YOLO polygon images."""
from contextlib import ExitStack
from pathlib import Path
import sys

import numpy as np

from .io.artifacts import write_run


TARGETS = {"stone": 1, "fiber": 2}


def target_masks(label_file, names, shape):
    import cv2

    height, width = shape
    masks = {name: np.zeros(shape, np.uint8) for name in TARGETS}
    for line_no, line in enumerate(label_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        xy = np.asarray(fields[1:], dtype=float)
        if (not 0 <= class_id < len(names) or xy.size < 6 or xy.size % 2
                or not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any()):
            raise ValueError(f"Invalid polygon in {label_file}, line {line_no}")
        name = names[class_id]
        if name in masks:
            points = np.rint(xy.reshape(-1, 2) * [width, height])
            points = np.clip(points, 0, [width - 1, height - 1]).astype(np.int32)
            cv2.fillPoly(masks[name], [points], 1)
    return masks


def counts(truth, predicted):
    return dict(reference=int(np.count_nonzero(truth)), predicted=int(predicted.sum()),
                intersection=int(np.count_nonzero(truth & predicted)))


def summarize(samples):
    summary = {}
    for name in TARGETS:
        rows = [sample[name] for sample in samples]
        total = sum(r["reference"] + r["predicted"] for r in rows)
        intersection = sum(r["intersection"] for r in rows)
        empty = [r for r in rows if r["reference"] == 0]
        summary[name] = dict(
            micro_dice=2 * intersection / total if total else None,
            micro_iou=intersection / (total - intersection) if total else None,
            empty_reference=len(empty),
            false_positive_images=sum(r["predicted"] > 0 for r in empty),
        )
    return summary


def evaluate_images(dataset: Path, splits, uretersam: Path, output: Path, device: str):
    import cv2
    import torch
    import yaml
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(uretersam.resolve()))
    from g0_runtime import Runtime
    from g0_runtime.assets import MODEL_MANIFEST, load_models, read, seed_everything, sha256
    from g0_runtime.runner import source_hashes
    from tools.fiber_photometric_input import infer

    names = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    if not isinstance(names, list) or any(names.count(name) != 1 for name in TARGETS):
        raise ValueError(f"Expected unique stone and fiber classes in {dataset}/data.yaml")
    if len(set(splits)) != len(splits):
        raise ValueError("Repeated split")
    seed_everything()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    experts = load_models(device)
    runtime = Runtime()
    manifest = dict(dataset=dataset.name, names=names, models=read(MODEL_MANIFEST),
                    data_yaml_sha256=sha256(dataset / "data.yaml"), device=device,
                    runtime_source_sha256=source_hashes(), mode="independent_image_g0",
                    splits={}, torch_version=str(torch.__version__), cv2_version=cv2.__version__)
    with ExitStack() as stack:
        write = stack.enter_context(write_run(output, manifest))
        stack.enter_context(torch.inference_mode())
        stack.enter_context(threadpool_limits(limits=1))
        for split in splits:
            directory = dataset / split
            images = sorted(p for p in (directory / "images").iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
            labels = {p.stem for p in (directory / "labels").glob("*.txt")}
            if not images or len(labels) != len(images) or labels != {p.stem for p in images}:
                raise ValueError(f"Missing, duplicate, or unmatched images/labels in {directory}")
            samples = []
            for image in images:
                frame = cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Unreadable image {image}")
                label_file = directory / "labels" / (image.stem + ".txt")
                truth = target_masks(label_file, names, frame.shape[:2])
                runtime.reset()
                raw, _, prob, _ = infer(experts, frame, "raw2")
                runtime.run(frame, raw, prob, 0.0, image.name)
                prediction = runtime.g0.final_mask
                if prediction.shape != frame.shape[:2] or not np.isin(prediction, [0, 1, 2]).all():
                    raise ValueError(f"Invalid predicted mask for {image}")
                result = {name: counts(truth[name], prediction == index)
                          for name, index in TARGETS.items()}
                write(dict(split=split, image=image.name, image_sha256=sha256(image),
                           label_sha256=sha256(label_file), **result))
                samples.append(result)
            manifest["splits"][split] = dict(images=len(samples), **summarize(samples))
            print(f"{dataset.name} {split}: {len(samples)} images", flush=True)
    return manifest["splits"]
