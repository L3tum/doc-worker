"""
Doc-Worker — Unit tests for serialization.json_default.

json_default is the shared last-chance numpy→JSON backstop used by both
server.SafeJSONResponse and the worker sidecar (json.dump default=).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from serialization import json_default


class TestJsonDefault:
    def test_ndarray(self):
        assert json_default(np.array([1, 2], dtype=np.int32)) == [1, 2]

    def test_numpy_scalar(self):
        out = json_default(np.float32(0.5))
        assert out == 0.5
        assert type(out) is float

    def test_item_fallback(self):
        class _ItemOnly:
            def item(self) -> float:
                return 3.5

        assert json_default(_ItemOnly()) == 3.5

    def test_unknown_type_raises(self):
        """No silent str() fallback: unknown types must fail loudly."""
        with pytest.raises(TypeError):
            json_default(object())

    def test_not_called_for_dict_keys(self):
        """documented limitation: default= is never consulted for dict keys.

        Pins the current behavior so nobody "fixes" json_default assuming
        it converts keys — it cannot (json.dumps calls it for values only).
        PaddleX payload keys are strings, so this does not bite in practice.
        """

        class _Key:
            pass

        with pytest.raises(TypeError):
            json.dumps({_Key(): "value"}, default=json_default)

    def test_nan_values_not_handled_by_default(self):
        """documented limitation: non-finite values are not sanitized here.

        Callers using allow_nan=False must sanitize upstream
        (paddlex_helpers._finite_confidence) — with a plain json.dumps the
        value simply serializes (no default= call for native floats).
        """
        out = json.dumps({"score": float("nan")}, default=json_default)
        assert "NaN" in out
