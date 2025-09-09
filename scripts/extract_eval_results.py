#!/usr/bin/env python3
"""
Extract evaluation results from a training log.

Rules:
- "全量测评结果": lines starting with "[Eval][Epoch n]" where n is a multiple of 3 (3, 6, 9, ...)
- "快速测评结果": lines starting with "[Eval][Epoch m]" where m is NOT a multiple of 3 (1, 2, 4, 5, 7, ...)
- Epoch 0 must appear in BOTH categories.
- Save both categories into a single txt file.

Usage:
  python dagr/scripts/extract_eval_results.py \
      --input /Users/sunxichen/Product/spike-project/dagr/snn_yaml_s_fasttrend_20250904_212820.log \
      --output /Users/sunxichen/Product/spike-project/dagr/snn_yaml_s_fasttrend_20250904_212820_extracted.txt

If --input/--output are omitted, sensible defaults are used (input points to the known log, output
is created alongside it with suffix "_extracted.txt").
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


EVAL_EPOCH_PATTERN = re.compile(r"^\s*\[Eval\]\[Epoch\s+(\d+)\].*")


def parse_arguments() -> argparse.Namespace:
    default_input = \
        "/Users/sunxichen/Product/spike-project/dagr/snn_yaml_s_fasttrend_20250904_212820.log"
    default_output = str(Path(default_input).with_name(
        Path(default_input).stem + "_extracted.txt"
    ))

    parser = argparse.ArgumentParser(
        description=(
            "Extract '[Eval][Epoch n]' lines from a log into full and fast evaluation sections."
        )
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        default=default_input,
        help="Path to the input log file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=str,
        default=default_output,
        help="Path to the output txt file.",
    )
    return parser.parse_args()


def classify_eval_line(line: str) -> tuple[int | None, str]:
    """Return (epoch, line) if the line is an [Eval][Epoch N] line, else (None, line)."""
    match = EVAL_EPOCH_PATTERN.match(line)
    if not match:
        return None, line
    try:
        epoch = int(match.group(1))
    except ValueError:
        return None, line
    return epoch, line.rstrip("\n")


def extract_eval_sections(input_path: str) -> tuple[list[str], list[str]]:
    """Read the log and split into full and fast evaluation line lists.

    - Epoch 0 is included in both lists.
    - Epochs divisible by 3 go to full; others go to fast.
    """
    full_eval_lines: list[str] = []
    fast_eval_lines: list[str] = []

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            epoch, line = classify_eval_line(raw_line)
            if epoch is None:
                continue

            if epoch == 0:
                full_eval_lines.append(line)
                fast_eval_lines.append(line)
            elif epoch % 3 == 0:
                full_eval_lines.append(line)
            else:
                fast_eval_lines.append(line)

    return full_eval_lines, fast_eval_lines


def write_output(output_path: str, full_eval_lines: list[str], fast_eval_lines: list[str]) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("全量测评结果 (每三轮进行一次全量测评集上的评估)：\n")
        out.write("=" * 60 + "\n")
        for line in full_eval_lines:
            out.write(line + "\n")

        out.write("\n")
        out.write("快速测评结果 (在全量测评之间进行每轮次的快速测评，只使用1000大小的测评集)：\n")
        out.write("=" * 60 + "\n")
        for line in fast_eval_lines:
            out.write(line + "\n")


def main() -> None:
    args = parse_arguments()
    input_path = args.input_path
    output_path = args.output_path

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input log not found: {input_path}")

    full_eval_lines, fast_eval_lines = extract_eval_sections(input_path)
    write_output(output_path, full_eval_lines, fast_eval_lines)

    print(f"Written extracted results to: {output_path}")


if __name__ == "__main__":
    main()


