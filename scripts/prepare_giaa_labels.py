#!/usr/bin/env python3
"""Generate the derived GIAA label files for the PARA dataset.

The official PARA archive stores the GIAA annotations below
``PARA/annotation``. This script reads ``PARA-GiaaTrain.csv`` and
``PARA-GiaaTest.csv`` and converts their generic aesthetic rating statistics
into the soft-label distributions consumed by the GIAA data loader.

The distribution follows the score-distribution construction from DeQA-Score:
the normal density is integrated over regions centred at 1.0, 1.5, ..., 5.0,
then a linear post-adjustment solves the sum and expectation constraints. The
number and spacing of levels can be changed through --score-levels while the
default remains the nine levels used by the PARA release.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


SCORE_LEVELS = np.arange(1.0, 5.0 + 0.5, 0.5)
SCORE_COLUMNS = [f"aestheticScore_{score:.1f}" for score in SCORE_LEVELS]
REQUIRED_GIAA_ANNOTATIONS = {
    "PARA-GiaaTrain.csv": {
        "imageName",
        "sessionId",
        "aestheticScore_mean",
        "aestheticScore_std",
        *SCORE_COLUMNS,
    },
    "PARA-GiaaTest.csv": {
        "imageName",
        "sessionId",
        "aestheticScore_mean",
        "aestheticScore_std",
        *SCORE_COLUMNS,
    },
}


def _candidate_roots(source: Path) -> list[Path]:
    """Yield likely dataset roots for both extracted archive layouts.

    Depending on the extraction tool, users may pass either ``.../PARA`` or
    the parent directory containing a nested ``PARA`` directory.
    """

    source = source.expanduser().resolve()
    candidates = [source]
    if source.is_dir():
        candidates.append(source / "PARA")
        if source.name.lower() == "para":
            candidates.append(source.parent / "PARA")
    seen: set[Path] = set()
    resolved_candidates = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            resolved_candidates.append(candidate)
    return resolved_candidates


def resolve_dataset_root(source: str | os.PathLike[str]) -> Path:
    """Resolve an extracted PARA root and reject an archive path early."""

    source_path = Path(source).expanduser()
    if source_path.suffix.lower() == ".zip":
        raise ValueError(
            "--source points to a ZIP archive. Extract the archive first; "
            "the preparation step reads the extracted GIAA annotations."
        )
    for candidate in _candidate_roots(source_path):
        annotation = candidate / "annotation"
        if annotation.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find PARA/annotation below {source_path}. Expected an "
        "extracted PARA directory or a parent containing PARA/."
    )


def _validate_csv(path: Path, required_columns: set[str]) -> None:
    """Validate headers without loading a potentially large CSV into memory."""

    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # pragma: no cover - pandas supplies the details
        raise ValueError(f"Could not read {path}: {exc}") from exc
    missing = sorted(required_columns - columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def validate_giaa_annotations(dataset_root: Path) -> dict[str, Path]:
    """Check the two original GIAA annotation files and return their paths."""

    annotation_dir = dataset_root / "annotation"
    paths: dict[str, Path] = {}
    for filename, required_columns in REQUIRED_GIAA_ANNOTATIONS.items():
        path = annotation_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required PARA annotation: {path}")
        _validate_csv(path, required_columns)
        paths[filename] = path
    return paths


def _linear_interpolation_distribution(mean: float, score_levels: np.ndarray) -> np.ndarray:
    """Return a two-point distribution for a near-zero variance row."""

    result = np.zeros(len(score_levels), dtype=float)
    if not np.isfinite(mean):
        result[len(score_levels) // 2] = 1.0
        return result
    if mean <= score_levels[0]:
        result[0] = 1.0
        return result
    if mean >= score_levels[-1]:
        result[-1] = 1.0
        return result

    right = int(np.searchsorted(score_levels, mean, side="right"))
    left = right - 1
    width = score_levels[right] - score_levels[left]
    result[left] = (score_levels[right] - mean) / width
    result[right] = (mean - score_levels[left]) / width
    return result


def _counts_distribution(
    counts: np.ndarray,
    means: np.ndarray,
    score_levels: np.ndarray,
) -> np.ndarray:
    """Return a safe empirical fallback for invalid/degenerate rows."""

    counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
    counts = np.maximum(counts, 0.0)
    totals = counts.sum(axis=1, keepdims=True)
    result = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    empty = totals[:, 0] <= 0
    if empty.any():
        fallback_means = np.nan_to_num(
            means[empty],
            nan=float(score_levels.mean()),
            posinf=float(score_levels[-1]),
            neginf=float(score_levels[0]),
        )
        for row_index, fallback_mean in zip(np.flatnonzero(empty), fallback_means):
            result[row_index] = _linear_interpolation_distribution(fallback_mean, score_levels)
    return result


def _adjust_gaussian_distribution(
    mean: float,
    std: float,
    score_levels: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Build a soft label and solve the score-distribution constraints."""

    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-6:
        return _linear_interpolation_distribution(mean, score_levels), 1.0, 0.0

    step = float(np.median(np.diff(score_levels)))
    boundaries = np.concatenate(([score_levels[0] - step / 2.0], score_levels + step / 2.0))
    raw = ndtr((boundaries[1:] - mean) / std) - ndtr((boundaries[:-1] - mean) / std)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    level_count = len(score_levels)
    raw_sum = float(raw.sum())
    raw_mean = float(np.dot(raw, score_levels))
    level_mean = float(score_levels.mean())
    denominator = raw_mean - level_mean * raw_sum
    if abs(denominator) <= 1e-12:
        return _linear_interpolation_distribution(mean, score_levels), 1.0, 0.0

    alpha = (mean - level_mean) / denominator
    beta = (1.0 - alpha * raw_sum) / level_count
    adjusted = np.maximum(alpha * raw + beta, 0.0)
    if not np.isfinite(adjusted).all() or adjusted.sum() <= 0:
        return _linear_interpolation_distribution(mean, score_levels), 1.0, 0.0
    return adjusted, float(alpha), float(beta)


def discretize_scores(
    frame: pd.DataFrame,
    score_levels: np.ndarray | None = None,
) -> pd.DataFrame:
    """Convert raw GIAA annotations to the configured soft-label format."""

    score_levels = np.asarray(score_levels if score_levels is not None else SCORE_LEVELS, dtype=float)
    score_columns = [f"aestheticScore_{score:.1f}" for score in score_levels]
    output_score_columns = [f"aesScore_{score:.1f}" for score in score_levels]
    missing_columns = [column for column in score_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing score columns: {', '.join(missing_columns)}")

    counts = frame[score_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    means = pd.to_numeric(frame["aestheticScore_mean"], errors="coerce").to_numpy(float)
    stds = pd.to_numeric(frame["aestheticScore_std"], errors="coerce").to_numpy(float)

    probabilities = np.zeros((len(frame), len(score_levels)), dtype=float)
    alphas = np.ones(len(frame), dtype=float)
    betas = np.zeros(len(frame), dtype=float)
    fallback = _counts_distribution(counts, means, score_levels)
    for row_index, (mean, std) in enumerate(zip(means, stds)):
        if np.isfinite(mean) and np.isfinite(std) and std > 1e-6:
            probabilities[row_index], alphas[row_index], betas[row_index] = _adjust_gaussian_distribution(
                float(mean), float(std), score_levels
            )
        else:
            probabilities[row_index] = fallback[row_index]

    output_means = means.copy()
    output_stds = stds.copy()
    normalized_probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    invalid_stats = ~np.isfinite(output_means)
    output_means[invalid_stats] = normalized_probabilities[invalid_stats] @ score_levels
    invalid_std = ~np.isfinite(output_stds) | (output_stds < 0)
    output_stds[invalid_std] = np.sqrt(
        np.maximum(
            (normalized_probabilities[invalid_std] * score_levels**2).sum(axis=1)
            - output_means[invalid_std] ** 2,
            0.0,
        )
    )

    result = pd.DataFrame(
        {
            "imageName": frame["imageName"].astype(str).to_numpy(),
            "sessionId": frame["sessionId"].astype(str).to_numpy(),
        }
    )
    for index, column in enumerate(output_score_columns):
        result[column] = probabilities[:, index]
    result["aesScore_mean"] = [round(float(value), 2) for value in output_means]
    result["aesScore_std"] = [round(float(value), 2) for value in output_stds]
    result["alpha"] = alphas
    result["beta"] = betas
    return result


def prepare_giaa_labels(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    score_levels: np.ndarray | None = None,
) -> Path:
    """Generate train/test GIAA soft-label CSV files and return the output directory."""

    dataset_root = resolve_dataset_root(source)
    annotation_paths = validate_giaa_annotations(dataset_root)
    output_root = Path(output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for split, filename in (("train", "PARA-GiaaTrain.csv"), ("test", "PARA-GiaaTest.csv")):
        frame = pd.read_csv(annotation_paths[filename])
        derived = discretize_scores(frame, score_levels=score_levels)
        derived.to_csv(
            output_root / f"para_giaa_{split}_discretized.csv",
            index=False,
            float_format="%.9f",
        )

    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Extracted PARA directory, or its parent directory containing PARA/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory receiving the two derived GIAA label files.",
    )
    parser.add_argument(
        "--score-levels",
        default=",".join(f"{score:.1f}" for score in SCORE_LEVELS),
        help="Comma-separated score centers; nine PARA levels are the default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = prepare_giaa_labels(
            source=args.source,
            output=args.output,
            score_levels=np.fromstring(args.score_levels, sep=",")
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"prepare_giaa_labels.py: error: {exc}", file=sys.stderr)
        return 2
    print(f"Generated GIAA label files in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
