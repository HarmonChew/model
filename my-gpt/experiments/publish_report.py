"""Publish a stage report containing a log's verbatim program output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence


RUNNER_FIELDS = {
    "__RUN_TIME_SECONDS__": "Run time (seconds)",
    "__DEVICE__": "Device",
    "__GPU_MAX_UTIL__": "Peak GPU utilization",
    "__GPU_MAX_MEM__": "Peak GPU memory",
    "__EXIT_CODE__": "Exit code",
    "__RUN_OK__": "Run status",
}
RUNNER_PREFIXES = ("__RUN_", "__DEVICE__", "__GPU_", "__EXIT_CODE__")


@dataclass(frozen=True)
class CapturedOutput:
    stdout: str
    metadata: tuple[tuple[str, str], ...]


def split_runner_metadata(log_text: str) -> CapturedOutput:
    """Separate a valid trailing runner block without changing stdout text."""

    lines = log_text.splitlines(keepends=True)
    end = len(lines) - 1
    while end >= 0 and not lines[end].rstrip("\r\n"):
        end -= 1
    if end < 0 or not lines[end].rstrip("\r\n").startswith(RUNNER_PREFIXES):
        return CapturedOutput(log_text, ())

    start = end
    while start > 0:
        previous = lines[start - 1].rstrip("\r\n")
        if not previous or not previous.startswith(RUNNER_PREFIXES):
            break
        start -= 1

    metadata: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in lines[start : end + 1]:
        stripped = line.rstrip("\r\n")
        if not stripped:
            continue
        marker, separator, value = stripped.partition(" ")
        if marker not in RUNNER_FIELDS:
            if stripped.startswith(RUNNER_PREFIXES):
                raise ValueError(f"unknown runner metadata field: {marker}")
            raise ValueError("runner metadata must be a contiguous trailing block")
        if marker in seen:
            raise ValueError(f"duplicate runner metadata field: {marker}")
        seen.add(marker)
        if marker == "__RUN_OK__":
            if separator and value:
                raise ValueError("__RUN_OK__ does not accept a value")
            value = "ok"
        elif not separator or not value:
            raise ValueError(f"runner metadata field has no value: {marker}")
        metadata.append((RUNNER_FIELDS[marker], value))

    return CapturedOutput("".join(lines[:start]), tuple(metadata))


def markdown_fence(stdout: str) -> str:
    """Choose a fence longer than every backtick run in stdout."""

    longest = max((len(match.group(0)) for match in re.finditer(r"`+", stdout)), default=0)
    return "`" * max(3, longest + 1)


def render_report(stage: int, script: str, captured: CapturedOutput) -> str:
    if stage < 1:
        raise ValueError("stage must be a positive integer")
    if not script:
        raise ValueError("script must not be empty")

    lines = [
        f"# Stage {stage} script output\n",
        "\n",
        f"- Script: `{script}`\n",
    ]
    for label, value in captured.metadata:
        lines.append(f"- {label}: `{value}`\n")
    lines.extend(["\n", "## Stdout\n", "\n"])

    fence = markdown_fence(captured.stdout)
    lines.append(f"{fence}text\n")
    lines.append(captured.stdout)
    if captured.stdout and not captured.stdout.endswith(("\n", "\r")):
        lines.append("\n")
    lines.append(f"{fence}\n")
    return "".join(lines)


def publish_report(
    *, stage: int, script: str, log_path: Path, output_path: Path, force: bool = False
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"report already exists: {output_path}")
    log_text = log_path.read_text(encoding="utf-8")
    report = render_report(stage, script, split_runner_metadata(log_text))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8", newline="")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    publish_report(
        stage=args.stage,
        script=args.script,
        log_path=args.log,
        output_path=args.output,
        force=args.force,
    )


if __name__ == "__main__":
    main()
