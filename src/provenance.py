"""Provenance fields every results file must carry (PLAN.md guardrails 4-5)."""

import datetime
import pathlib
import platform
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent


def provenance(seed):
    """seed, git_sha, timestamp (+ whether the tree was clean) for a results file."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return {
        "seed": seed,
        "git_sha": sha,
        "git_dirty": dirty,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "machine": platform.platform(),
        "python": platform.python_version(),
    }
