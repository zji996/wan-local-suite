"""Compatibility shim for the Wan2.2 third-party package."""

from __future__ import annotations

import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_source_dir = _pkg_dir.parent / "wan2.2"
if not _source_dir.exists():
    raise ImportError(f"Wan2.2 sources not found at {_source_dir}")

__path__ = [str(_source_dir)]

# Allow legacy imports like `third_party.wan2.2.*` to reuse this module.
sys.modules.setdefault("third_party.wan2.2", sys.modules[__name__])
