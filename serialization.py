"""
Doc-Worker — shared JSON serialization helpers
==============================================

Single home for the numpy→JSON safety policy, so the server responses
(SafeJSONResponse in server.py) and the worker sidecar (json.dump in
worker.py) cannot drift apart.

Primary conversion happens at the source (paddlex_helpers._jsonable);
json_default is the last line of defense for non-JSON-native *types* that
ever leak into a payload. Note it handles values, not keys: non-str dict
keys still raise (documented limitation — PaddleX payload keys are strings).
"""

from __future__ import annotations


def json_default(obj: object) -> object:
    """json default= fallback: convert numpy types to Python equivalents.

    Handles numpy arrays/scalars (via .tolist()) and other objects that
    expose an .item(). Raises TypeError for anything else so json.dumps
    keeps failing loudly on truly unknown types (no silent str() fallback).

    Limitations (by design):
      - Non-JSON-native *dict keys* are not converted (json.dumps calls
        default= only for values). PaddleX payload keys are strings, so
        this does not bite in practice — do not "fix" by assuming it does.
      - Non-finite float *values* (NaN/Inf) are not handled here; callers
        that serialize with allow_nan=False must sanitize values upstream
        (see paddlex_helpers._finite_confidence).
    """
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
