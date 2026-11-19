"""Pilot supervised depth-head adaptation with training-only scale calibration."""
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np

from .depth_benchmark import SingleFrameModel, partition
from .contracts import DepthObservation
from .io.artifacts import write_run


def _log_depth(model, rgb):
    import torch

    pred = model.forward(rgb)
    if model.model_id == "v2-small":
        return -torch.log1p(pred.clamp_min(0))
    return pred.clamp_min(1e-6).log()


def _decoder(model, scope):
    import torch

    if model.model_id == "v2-small" and scope == "decoder":
        return torch.nn.ModuleDict({"neck": model.net.neck, "head": model.net.head})
    return model.net.head


class AdaptedDepth:
    """Replay adapted native-depth estimates before metric camera validation."""

    def __init__(self, weights, run, device):
        from safetensors.torch import load_file

        meta = json.loads((run / "manifest.json").read_text())
        if meta.get("target") != "native_tiff":
            raise ValueError("Unknown adapted depth convention")
        if meta.get("weights_sha256") != hashlib.sha256((weights / "model.safetensors").read_bytes()).hexdigest():
            raise ValueError("Base weights differ from the fine-tuning run")
        self.model = SingleFrameModel(meta["model"], weights, device)
        decoder = _decoder(self.model, meta.get("scope", "head"))
        decoder.load_state_dict(load_file(str(run / "depth_head.safetensors")))
        self.slope = meta["log_slope"]
        self.intercept = meta["log_intercept"]
        self.source = meta["model"] + "_kidneydepth_native"

    def estimate(self, frame):
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            log_z = _log_depth(self.model, frame.image_bgr[..., ::-1])
            z = (self.intercept + self.slope * log_z).exp()
            out = F.interpolate(z[:, None], frame.shape, mode="bilinear", align_corners=False)
        values = out[0, 0].cpu().numpy()
        return DepthObservation(values, np.isfinite(values) & (values > 0),
                                "relative_depth", frame.pixel_domain, self.source)


def _sample(groups, split, count, rng=None):
    rows = []
    for (side, trajectory), pairs in groups.items():
        if side == split:
            n = min(count, len(pairs))
            indices = (np.linspace(0, len(pairs)-1, n, dtype=int) if rng is None
                       else rng.choice(len(pairs), n, replace=False))
            rows.extend((trajectory, *pairs[i]) for i in indices)
    if rng is not None:
        rng.shuffle(rows)
    return rows


def _read(archive, image, depth, device):
    from PIL import Image
    import tifffile
    import torch

    rgb = np.asarray(Image.open(BytesIO(archive.read(image))).convert("RGB"))
    truth = tifffile.imread(BytesIO(archive.read(depth))).astype(np.float32)
    if rgb.shape[:2] != truth.shape:
        raise ValueError(f"Misaligned depth for {image}")
    return rgb, torch.from_numpy(truth).to(device)


def evaluate_adapted(archive_path, weights, run, output, per_sequence, device):
    """Compare scale-calibrated frozen and adapted predictions on identical test pixels."""
    import torch
    import torch.nn.functional as F

    if per_sequence < 1:
        raise ValueError("Samples per trajectory must be positive")
    adapted = AdaptedDepth(weights, run, device)
    frozen = SingleFrameModel(adapted.model.model_id, weights, device)
    summary = {}
    manifest = dict(split="test", depth_run=str(run.resolve()), target="native_tiff",
                    per_sequence=per_sequence, summary=summary)
    with ZipFile(archive_path) as archive, write_run(output, manifest) as write, torch.no_grad():
        rows = _sample(partition(archive), "test", per_sequence)
        sequences = defaultdict(list)
        for seq, im, dep in rows:
            rgb, target = _read(archive, im, dep, device)
            valid = torch.isfinite(target) & (target > 0)
            scores = {}
            for name, model in [("frozen", frozen), ("adapted", adapted.model)]:
                z = (adapted.intercept + adapted.slope * _log_depth(model, rgb)).exp()
                z = F.interpolate(z[:, None], target.shape, mode="bilinear", align_corners=False)[0, 0]
                score = ((z[valid] - target[valid]).abs() / target[valid]).mean().item()
                if not np.isfinite(score):
                    raise ValueError(f"Invalid depth error for {im}")
                scores[name] = score
            write(dict(image=im, **scores))
            sequences[seq].append(scores)
        for name in ("frozen", "adapted"):
            summary[name + "_abs_rel"] = float(np.mean([
                np.mean([r[name] for r in records]) for records in sequences.values()]))
    return summary


def finetune(archive_path, model_id, weights, output, epochs, samples, device):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file

    if min(epochs, samples) < 1:
        raise ValueError("Positive epochs and samples per sequence required")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = SingleFrameModel(model_id, weights, device)
    model.net.requires_grad_(False)
    head = _decoder(model, "decoder").requires_grad_(True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0001)
    checkpoint = output / "depth_head.safetensors"
    with ZipFile(archive_path) as archive:
        groups = partition(archive)
        val = _sample(groups, "valid", 4)
        test = _sample(groups, "test", 4)

        # Global affine mapping in log depth, fitted only to the training trajectories.
        x, y = [], []
        with torch.no_grad():
            for _, im, dep in _sample(groups, "train", 1):
                rgb, truth = _read(archive, im, dep, device)
                obs = _log_depth(model, rgb)
                target = F.interpolate(truth[None, None], obs.shape[-2:], mode="nearest")[0, 0]
                target, obs = target[::16, ::16], obs[0, ::16, ::16]
                valid = torch.isfinite(target) & (target > 0) & torch.isfinite(obs)
                x.extend(obs[valid].tolist())
                y.extend(target[valid].log().tolist())
        slope, intercept = np.polyfit(x, y, 1).tolist()
        if not np.isfinite([slope, intercept]).all() or slope <= 0:
            raise ValueError("Training calibration failed to preserve the depth ordering")

        @torch.no_grad()
        def evaluate(rows):
            by_sequence = defaultdict(list)
            for seq, im, dep in rows:
                rgb, target = _read(archive, im, dep, device)
                z = torch.exp(intercept + slope * _log_depth(model, rgb))
                z = F.interpolate(z[:, None], target.shape, mode="bilinear", align_corners=False)[0, 0]
                valid = torch.isfinite(target) & (target > 0)
                error = ((z[valid] - target[valid]).abs() / target[valid]).mean().item()
                if not np.isfinite(error):
                    raise ValueError(f"Invalid validation prediction in {im}")
                by_sequence[seq].append(error)
            return float(np.mean([np.mean(v) for v in by_sequence.values()]))

        before = evaluate(val)
        best = before
        best_epoch = 0
        history = [before]

        def save_head():
            save_file({k: t.detach().cpu().contiguous().clone() for k, t in head.state_dict().items()},
                      str(checkpoint))

        save_head()
        for epoch in range(1, epochs + 1):
            rows = _sample(groups, "train", samples, rng)
            for _, im, dep in rows:
                rgb, target = _read(archive, im, dep, device)
                pred = intercept + slope * _log_depth(model, rgb)
                ref = F.interpolate(target[None, None], pred.shape[-2:], mode="nearest")[:, 0]
                valid = torch.isfinite(ref) & (ref > 0)
                loss = F.smooth_l1_loss(pred[valid], ref[valid].log(), beta=.1)
                if not torch.isfinite(loss):
                    raise ValueError(f"Invalid training loss in {im}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
            error = evaluate(val)
            history.append(error)
            if error < best:
                best, best_epoch = error, epoch
                save_head()
            print(f"Epoch {epoch}: validation AbsRel {error:.5f}", flush=True)
        head.load_state_dict(load_file(str(checkpoint)))
        test_after = evaluate(test)
        summary = dict(model=model_id, weights=str(weights.resolve()), seed=42,
                       scope="decoder", trainable_parameters=sum(p.numel() for p in head.parameters()),
                       weights_sha256=hashlib.sha256((weights / "model.safetensors").read_bytes()).hexdigest(),
                       input_size=518, epochs=epochs, samples_per_train_sequence=samples,
                       steps_per_epoch=len(rows), validation_frames=len(val), test_frames=len(test),
                       log_slope=slope, log_intercept=intercept, target="native_tiff",
                       validation_abs_rel=history, selected_epoch=best_epoch,
                       selected_test_abs_rel=test_after)
        (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
