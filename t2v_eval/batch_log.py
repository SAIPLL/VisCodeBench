"""
Batch T2V logging for VisCodeBench submissions.

Runs every completed ``<sample_id>.py`` under a data directory with the T2V
logging hooks, and collects the resulting logs into the folder layout that the
Codabench leaderboard expects::

    data/<subset>/<sample_id>/<sample_id>.py      (input, prediction filled in)
    logs/<subset>/<sample_id>/series.jsonl        (raw log)
    submission/<subset>/<sample_id>.jsonl         (file to submit)

Usage
-----
    python -m t2v_eval.batch_log --data_dir data --zip

Each script is executed in a subprocess with its sample directory as the
working directory. These scripts contain model-generated code: run this in an
isolated environment (container, VM or throwaway account).
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from t2v_eval.logging import LoggingState, prepare_py_file_for_logging

SERIES_JSONL = "series.jsonl"
INSTRUMENTED_SUFFIX = ".t2v_log.py"


def find_samples(data_dir: str, subsets: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Return ``(subset, sample_id, py_path)`` for every ``<subset>/<id>/<id>.py``."""
    samples = []
    for sample_dir in sorted(glob.glob(os.path.join(data_dir, "*", "*", ""))):
        sample_dir = sample_dir.rstrip(os.sep)
        sample_id = os.path.basename(sample_dir)
        subset = os.path.basename(os.path.dirname(sample_dir))
        if subsets and subset not in subsets:
            continue
        py_path = os.path.join(sample_dir, f"{sample_id}.py")
        if os.path.isfile(py_path):
            samples.append((subset, sample_id, py_path))
    return samples


def has_series(log_dir: str) -> bool:
    """True if ``log_dir/series.jsonl`` exists and is non-empty."""
    path = os.path.join(log_dir, SERIES_JSONL)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def log_sample(py_path: str,
               log_dir: str,
               python: str,
               timeout: int,
               keep_instrumented: bool = False) -> tuple[LoggingState, str]:
    """
    Run one script with T2V logging and return ``(state, message)``.

    The logging activation line is inserted just before
    ``##START VISUALISATION CODE`` in a temporary copy of the script, which is
    run from the sample directory so relative data paths resolve.
    """
    sample_dir = os.path.dirname(os.path.abspath(py_path))
    instrumented = os.path.abspath(py_path)[: -len(".py")] + INSTRUMENTED_SUFFIX

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    if prepare_py_file_for_logging(py_path, instrumented, os.path.abspath(log_dir)) is None:
        return LoggingState.BAD_INPUT, "'##START VISUALISATION CODE' marker not found"

    env = dict(os.environ, MPLBACKEND="Agg")  # never open a GUI window
    message = ""
    try:
        result = subprocess.run(
            [python, os.path.basename(instrumented)],
            cwd=sample_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode != 0:
            tail = result.stderr.strip().splitlines()[-1:] or [""]
            message = f"exit code {result.returncode}: {tail[0]}"
    except subprocess.TimeoutExpired:
        message = f"timeout after {timeout}s"
    finally:
        if not keep_instrumented and os.path.exists(instrumented):
            os.remove(instrumented)

    # A script that crashes after plotting may still have produced a usable log.
    if has_series(log_dir):
        return LoggingState.HAVING_DATA, message
    return LoggingState.NO_DATA, message or "script ran but plotted nothing"


def run(data_dir: str,
        log_root: str = "logs",
        submission_dir: str = "submission",
        subsets: list[str] | None = None,
        python: str = sys.executable,
        timeout: int = 300,
        workers: int = 1,
        skip_existing: bool = False,
        keep_instrumented: bool = False,
        make_zip: bool = False) -> list[dict]:
    samples = find_samples(data_dir, subsets)
    print(f"Found {len(samples)} samples under {data_dir}")

    def process(sample):
        subset, sample_id, py_path = sample
        log_dir = os.path.join(log_root, subset, sample_id)
        out_path = os.path.join(submission_dir, subset, f"{sample_id}.jsonl")
        t0 = time.time()
        if skip_existing and has_series(log_dir):
            state, message = LoggingState.HAVING_DATA, "cached"
        else:
            state, message = log_sample(py_path, log_dir, python, timeout, keep_instrumented)

        if state == LoggingState.HAVING_DATA:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copyfile(os.path.join(log_dir, SERIES_JSONL), out_path)
        elif os.path.exists(out_path):
            os.remove(out_path)  # don't submit a stale log from an earlier run

        return {
            "subset": subset,
            "sample_id": sample_id,
            "state": state.value,
            "duration_s": round(time.time() - t0, 2),
            "message": message,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(process, s) for s in samples]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            flag = "ok  " if row["state"] == LoggingState.HAVING_DATA.value else "FAIL"
            print(f"[{i}/{len(samples)}] {flag} {row['subset']}/{row['sample_id']}"
                  + (f"  ({row['message']})" if flag == "FAIL" else ""))

    rows.sort(key=lambda r: (r["subset"], r["sample_id"]))
    os.makedirs(log_root, exist_ok=True)
    report = os.path.join(log_root, "logging_report.csv")
    with open(report, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subset", "sample_id", "state", "duration_s", "message"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nSummary (samples with a log / total):")
    for subset in sorted({r["subset"] for r in rows}):
        sub = [r for r in rows if r["subset"] == subset]
        ok = sum(r["state"] == LoggingState.HAVING_DATA.value for r in sub)
        print(f"  {subset:16s} {ok}/{len(sub)}")
    print(f"Report: {report}")
    print(f"Submission files: {submission_dir}/")

    if make_zip:
        zip_path = os.path.abspath(submission_dir).rstrip(os.sep) + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(glob.glob(os.path.join(submission_dir, "*", "*.jsonl"))):
                zf.write(path, os.path.relpath(path, submission_dir))
        print(f"Zipped: {zip_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run T2V logging on completed VisCodeBench samples and build the submission folder.")
    parser.add_argument("--data_dir", required=True,
                        help="Folder containing the subset folders (e.g. data).")
    parser.add_argument("--log_dir", default="logs",
                        help="Where raw logs and the report are written (default: logs).")
    parser.add_argument("--submission_dir", default="submission",
                        help="Where <subset>/<sample_id>.jsonl files are written (default: submission).")
    parser.add_argument("--subsets", nargs="+",
                        help="Only process these subsets (default: all found).")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter that runs the scripts; it must have t2v_eval "
                             "and the plotting libraries installed (default: this interpreter).")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-sample timeout in seconds (default: 300).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of samples to run in parallel (default: 1).")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Reuse logs that already have a series.jsonl.")
    parser.add_argument("--keep_instrumented", action="store_true",
                        help=f"Keep the temporary <sample_id>{INSTRUMENTED_SUFFIX} files for debugging.")
    parser.add_argument("--zip", action="store_true",
                        help="Also write <submission_dir>.zip, ready to upload.")
    args = parser.parse_args()

    run(data_dir=args.data_dir,
        log_root=args.log_dir,
        submission_dir=args.submission_dir,
        subsets=args.subsets,
        python=args.python,
        timeout=args.timeout,
        workers=args.workers,
        skip_existing=args.skip_existing,
        keep_instrumented=args.keep_instrumented,
        make_zip=args.zip)


if __name__ == "__main__":
    main()
