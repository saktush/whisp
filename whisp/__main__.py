"""CLI for the whisp driver.

Replaces the `whisperx` CLI call in whisp.sh. whisperx's own CLI takes a
single --device for every stage, which forces diarization onto the CPU
because CTranslate2 cannot use Metal; this driver picks a device per stage
and overlaps the two that do not depend on each other.
"""

import argparse
import os
import pathlib
import sys

from whisp import devices, pipeline


def parse_args(argv: list[str], env: dict | None = None) -> argparse.Namespace:
    env = os.environ if env is None else env

    parser = argparse.ArgumentParser(prog="whisp", description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--language", default=env.get("WHISP_LANG", "ru"))
    parser.add_argument("--model", default=env.get("WHISP_MODEL", "turbo"))
    parser.add_argument("--compute-type", default=env.get("WHISP_COMPUTE_TYPE", "int8"))
    parser.add_argument("--device", default=env.get("WHISP_DEVICE", "auto"))
    parser.add_argument("--batch-size", type=int, default=int(env.get("WHISP_BATCH_SIZE", "8")))
    parser.add_argument(
        "--diarize-batch-size",
        type=int,
        default=int(env.get("WHISP_DIARIZE_BATCH_SIZE", "64")),
    )
    parser.add_argument("--threads", type=int, default=int(env.get("WHISP_ASR_THREADS", "0")))
    parser.add_argument("--no-diarize", dest="diarize", action="store_false")
    parser.set_defaults(diarize=True, parallel=env.get("WHISP_PARALLEL", "1") != "0")

    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = str(pathlib.Path(args.audio).resolve().parent)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    hf_token = os.environ.get("HF_TOKEN", "")
    if args.diarize and not hf_token:
        print("whisp: HF_TOKEN is not set; diarization needs it", file=sys.stderr)
        return 1

    device = devices.select_device(args.device)
    if device == "cpu" and args.device in (None, "auto"):
        print("whisp: MPS unavailable, running diarization and alignment on CPU", flush=True)

    pipeline.run(
        audio_path=args.audio,
        output_dir=args.output_dir,
        language=args.language,
        model_name=args.model,
        compute_type=args.compute_type,
        device=device,
        hf_token=hf_token,
        batch_size=args.batch_size,
        diarize_batch_size=args.diarize_batch_size,
        asr_thread_count=devices.asr_threads(args.threads),
        diarize_enabled=args.diarize,
        parallel=args.parallel,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
