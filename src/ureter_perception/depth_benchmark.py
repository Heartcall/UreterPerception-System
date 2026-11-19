"""Trajectory-disjoint, single-image Depth Anything comparison on KidneyDepthMetric."""
from collections import defaultdict
from io import BytesIO
import json
import re
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from .io.artifacts import write_run


def partition(archive):
    """Keep trajectory number, all lighting/materials and random variants together."""
    names = archive.namelist()
    pattern = re.compile(r"KidneyDepthMetric/([^/]+/[^/]+/[^/]+/traj_(?:random_)?(\d+))/images/(\d+)\.png")
    depth_names = {n for n in names if "/depth/" in n and n.endswith(".tif")}
    groups = defaultdict(list)
    for name in sorted(n for n in names if "/images/" in n and n.endswith(".png")):
        match = pattern.fullmatch(name)
        if match is None:
            raise ValueError(f"Unknown trajectory layout: {name}")
        trajectory, number, _ = match.groups()
        split = {"1": "train", "2": "train", "3": "valid", "4": "valid",
                 "5": "test", "6": "test"}.get(number)
        if split is None:
            raise ValueError(f"Unassigned trajectory: {trajectory}")
        label = name.replace("/images/", "/depth/")[:-4] + ".tif"
        if label not in depth_names:
            raise ValueError(f"Missing or duplicate pair: {name}")
        depth_names.remove(label)
        groups[split, trajectory].append((name, label))
    if depth_names or {s for s, _ in groups} != {"train", "valid", "test"}:
        raise ValueError("Missing partition or unpaired depth maps")
    return groups


def inventory(archive_path):
    with ZipFile(archive_path) as archive:
        groups = partition(archive)
    return {s: {t: len(items) for (split, t), items in groups.items() if split == s}
            for s in ("train", "valid", "test")}


def ordering_score(truth, prediction, inverse):
    """Pairwise ordering accuracy on a deterministic spatial sample, ties get half credit."""
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("Prediction and reference grids must agree")
    valid = np.isfinite(truth) & (truth > 0)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError("Insufficient valid reference")
    rng = np.random.default_rng(27)
    left, right = rng.choice(indices, size=(2, 8192))
    delta = truth.flat[left].astype(float) - truth.flat[right]
    comparable = delta != 0
    if not comparable.any():
        raise ValueError("No reference depth variation")
    delta = delta[comparable]
    left, right = left[comparable], right[comparable]
    diff = prediction.flat[left].astype(float) - prediction.flat[right]
    if inverse:
        diff = -diff
    finite = np.isfinite(diff)
    correct = np.sign(delta[finite]) == np.sign(diff[finite])
    credit = correct.sum() + 0.5 * np.count_nonzero(diff[finite] == 0)
    return float(credit / len(delta)), float(finite.mean())


class SingleFrameModel:
    """Use a common RGB grid and the published networks; preserve depth direction."""

    def __init__(self, model_id, weights, device):
        import torch

        if not weights.is_dir():
            raise FileNotFoundError(f"Missing local checkpoint directory: {weights}")
        self.model_id = model_id
        self.device = device
        if model_id == "v2-small":
            from transformers import AutoModelForDepthEstimation
            model = AutoModelForDepthEstimation.from_pretrained(weights, local_files_only=True)
            if model.config.backbone_config.hidden_size != 384 or model.config.model_type != "depth_anything":
                raise ValueError("Expected Depth Anything V2 Small HF checkpoint")
        elif model_id == "v3-small":
            from depth_anything_3.cfg import create_object, load_config
            from depth_anything_3.registry import MODEL_REGISTRY
            from safetensors.torch import load_model

            if json.loads((weights / "config.json").read_text())["model_name"] != "da3-small":
                raise ValueError("Expected DA3-Small checkpoint")
            model = create_object(load_config(MODEL_REGISTRY["da3-small"]))
            wrapped = torch.nn.Module()
            wrapped.add_module("model", model)
            load_model(wrapped, str(weights / "model.safetensors"), strict=True)
        else:
            raise ValueError(f"Unknown checkpoint type {model_id}")
        self.net = model.to(device).eval()
        self.mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

    def forward(self, rgb):
        import torch
        from PIL import Image

        im = Image.fromarray(rgb).resize((518, 518), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(im)).to(self.device).permute(2, 0, 1)[None].float()
        x = (x / 255 - self.mean) / self.std
        if self.model_id == "v2-small":
            return self.net(x).predicted_depth
        return self.net(x[:, None]).depth.reshape(1, 518, 518)


def load_predictor(model_id, weights, device):
    import torch
    import torch.nn.functional as F
    model = SingleFrameModel(model_id, weights, device)

    @torch.inference_mode()
    def predict(rgb):
        h, w = rgb.shape[:2]
        depth = model.forward(rgb).float()
        return F.interpolate(depth[:, None], (h, w), mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()
    return predict


def benchmark(archive_path, model_id, weights, output, per_sequence, device):
    import tifffile
    import torch
    from PIL import Image

    if per_sequence <= 0:
        raise ValueError("Samples per trajectory must be positive")
    predict = load_predictor(model_id, weights, device)
    summary = {}
    manifest = dict(model=model_id, weights=str(weights.resolve()), device=device,
                    split="valid", input_shape=[518, 518], samples_per_trajectory=per_sequence,
                    reference="native_tiff", summary=summary)
    scores, latency = [], []
    with ZipFile(archive_path) as archive, write_run(output, manifest) as write:
        for (split, trajectory), pairs in partition(archive).items():
            if split != "valid":
                continue
            series = []
            for i in np.linspace(0, len(pairs)-1, min(per_sequence, len(pairs)), dtype=int):
                image_name, label_name = pairs[i]
                rgb = np.asarray(Image.open(BytesIO(archive.read(image_name))).convert("RGB"))
                truth = tifffile.imread(BytesIO(archive.read(label_name)))
                if not latency:
                    predict(rgb)
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                start = perf_counter()
                depth = predict(rgb)
                if torch.device(device).type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = (perf_counter() - start) * 1000
                score, coverage = ordering_score(truth, depth, model_id == "v2-small")
                write(dict(image=image_name, order_accuracy=score, valid_pairs=coverage, ms=elapsed))
                series.append(score)
                latency.append(elapsed)
            scores.append(np.mean(series))
        summary.update(trajectory_mean_order_accuracy=float(np.mean(scores)),
                       p50_ms=float(np.median(latency)), p95_ms=float(np.percentile(latency, 95)))
    return summary
