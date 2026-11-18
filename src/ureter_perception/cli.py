import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

from .adapters.robot import RobotCsv
from .adapters.uretersam import UreterSAMReplay
from .alignment import RobotTimeline
from .datasets.kidneydepth import KidneyDepthSample
from .datasets.recording import Recording
from .depth import estimate_depth
from .geometry import observe_geometry
from .io.artifacts import json_array, write_run
from .io.scene import read_geometry, read_scene
from .pipeline import process_frame
from .segmentation import segment
from .targeting import measure


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, default=json_array))


def _video_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--g0-cache", type=Path, required=True)
    parser.add_argument("--limit", type=int)


def _source(args) -> UreterSAMReplay:
    layout = Recording(args.video.stem, args.video, args.g0_cache)
    return UreterSAMReplay(layout.video, layout.segmentation_dir, layout.stream_id)


def _replay(args) -> None:
    source = _source(args)
    with write_run(args.output, {
        "mode": "replay", "video_sha256": source.manifest["input_sha256"],
        "g0_cache": str(args.g0_cache.resolve()),
        "timestamp_source": "frame_index_fps", "depth_backend": None,
    }) as write:
        for cached in source.frames(args.limit):
            write(process_frame(cached.frame, cached).record())


def _synthetic(args) -> None:
    scene = read_scene(args.input)
    with write_run(args.output, {
        "mode": "synthetic", "depth_backend": "synthetic_truth",
        "calibration_id": scene.calibration.id,
    }) as write:
        write(process_frame(scene.frame, scene, scene, scene.calibration).record())


def _segmentation(args) -> None:
    count = 0
    observed_tips = 0
    for cached in _source(args).frames(args.limit):
        result = segment(cached.frame, cached)
        count += 1
        observed_tips += result.tip_uv is not None
    _print({"frames": count, "g0_tip_estimates": observed_tips})


def _geometry(args) -> None:
    scene = read_scene(args.input)
    geometry = observe_geometry(segment(scene.frame, scene), estimate_depth(scene.frame, scene), scene.calibration)
    _print(asdict(geometry))


def _targeting(args) -> None:
    _print(asdict(measure(read_geometry(args.input))))


def _depth(args) -> None:
    from .adapters.kidneydepth import read_metric_depth

    sample = KidneyDepthSample(args.trajectory, args.index)
    depth = read_metric_depth(args.archive, sample)
    _print({"image_member": sample.image_member, "depth_member": sample.depth_member,
            "shape": depth.values.shape, "finite_values": int(np.isfinite(depth.values).sum()),
            "source": "kidneydepth_dataset_truth", "unit": "unconverted_native_values"})


def _alignment(args) -> None:
    timeline = RobotTimeline(RobotCsv(args.input).samples(), args.clock_id)
    _print(asdict(timeline.at(args.capture_ns, args.decision_ns, args.max_age_ns)))


def _scene_args(parser: argparse.ArgumentParser, handler) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.set_defaults(handler=handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay and verify ureteroscopic perception modules")
    commands = parser.add_subparsers(required=True)
    run = commands.add_parser("run").add_subparsers(required=True)
    replay = run.add_parser("replay", help="Replay a UreterSAM video and matching cache")
    _video_args(replay)
    replay.add_argument("--output", type=Path, required=True)
    replay.set_defaults(handler=_replay)
    synthetic = run.add_parser("synthetic", help="Run a known metric scene")
    _scene_args(synthetic, _synthetic)
    synthetic.add_argument("--output", type=Path, required=True)

    module = commands.add_parser("eval").add_subparsers(required=True)
    segmentation = module.add_parser("segmentation")
    _video_args(segmentation)
    segmentation.set_defaults(handler=_segmentation)
    _scene_args(module.add_parser("geometry"), _geometry)
    _scene_args(module.add_parser("targeting"), _targeting)
    depth = module.add_parser("depth")
    depth.add_argument("--archive", type=Path, required=True)
    depth.add_argument("--trajectory", required=True)
    depth.add_argument("--index", type=int, default=0)
    depth.set_defaults(handler=_depth)
    alignment = module.add_parser("alignment")
    _scene_args(alignment, _alignment)
    alignment.add_argument("--clock-id", required=True)
    for arg in ("capture-ns", "decision-ns", "max-age-ns"):
        alignment.add_argument(f"--{arg}", type=int, required=True)

    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
