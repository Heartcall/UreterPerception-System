"""Evaluate the two saved S2 models on the published KidneyDepthSLAM sequences."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ureter_perception.depth_tuning import AdaptedDepth, _log_depth, _read
from ureter_perception.io.artifacts import write_run


def metrics(pred, gt):
    valid = torch.isfinite(gt) & (gt > 0)
    gt = gt[valid]
    pred = pred[valid]
    if not gt.numel() or not torch.all(torch.isfinite(pred) & (pred > 0)):
        raise ValueError("Empty reference or invalid prediction at scored pixels")
    difference = pred - gt
    ratio = torch.maximum(pred / gt, gt / pred)
    values = torch.stack([(difference.abs() / gt).mean(), (difference.abs() / pred).mean(),
                          difference.abs().mean(), difference.square().mean().sqrt(),
                          (pred.log() - gt.log()).square().mean().sqrt(),
                          *[(ratio < 1.25**i).float().mean() for i in (1, 2, 3)]])
    names = ["absrel", "absrel_pred_denominator", "mae_native", "rmse_native",
             "rmse_log", "delta1", "delta2", "delta3"]
    return dict(zip(names, values.tolist())), gt.numel()


@torch.inference_mode()
def evaluate(args):
    models = {}
    head_hashes = {}
    for name, base, run in [("v2", "Depth-Anything-V2-Small-hf", "v2-decoder"),
                            ("da3", "DA3-SMALL", "v3-small")]:
        tuned = args.assets / "finetuned" / run
        head_hashes[name] = hashlib.sha256((tuned / "depth_head.safetensors").read_bytes()).hexdigest()
        models[name] = AdaptedDepth(args.assets / "checkpoints" / base, tuned, args.device)
    summary = {}
    manifest = dict(dataset="KidneyDepthSLAM", stride=args.stride, head_sha256=head_hashes,
                    alignment="training calibration only", aggregation="mean of per-image metrics",
                    summary=summary)
    with ZipFile(args.archive) as archive, write_run(args.output, manifest) as write:
        names = archive.namelist()
        members = set(names)
        if len(names) != len(members):
            raise ValueError("Duplicate archive members")
        groups = defaultdict(list)
        for name in sorted(n for n in names if "/images/" in n and n.endswith(".png")):
            dep = name.replace("/images/", "/depth/")[:-4] + ".tif"
            if dep not in members:
                raise ValueError(f"Missing corresponding depth for {name}")
            groups[name.split("/")[1]].append((name, dep))
        if set(groups) != {f"Seq_{i:02}" for i in range(1, 6)}:
            raise ValueError("Expected five KidneyDepthSLAM sequences")
        total = defaultdict(list)
        for seq, pairs in sorted(groups.items()):
            series = defaultdict(list)
            for im, dep in pairs[::args.stride]:
                rgb, gt = _read(archive, im, dep, args.device)
                row = dict(image=im, sequence=seq)
                for name, backend in models.items():
                    depth = (backend.intercept + backend.slope * _log_depth(backend.model, rgb)).exp()
                    pred = F.interpolate(depth[:, None], gt.shape, mode="bilinear", align_corners=False)[0, 0]
                    values, pixels = metrics(pred, gt)
                    row[name] = values
                    row["valid_pixels"] = pixels
                    series[name].append(values)
                    total[name].append(values)
                write(row)
            summary[seq] = averages(series)
            print(seq, len(series["v2"]), flush=True)
        summary["all"] = averages(total)
        print(json.dumps(summary["all"]), flush=True)


def averages(groups):
    return {model: {key: float(np.mean([frame[key] for frame in frames]))
                    for key in frames[0]} for model, frames in groups.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    options = parser.parse_args()
    if options.stride < 1:
        parser.error("stride must be positive")
    evaluate(options)
