"""Import-hygiene tests — catch regressions that leaked in production.

These tests prevent two classes of bugs:

1. **Private PaddleX imports** — importing from ``paddlex._*`` modules (e.g.
   ``paddlex._utils.logging``) breaks when PaddleX renames or removes internal
   modules.  Use ``logging.getLogger("paddlex")`` instead.

2. **Wrong runtime-dependency checks** — the OCRmyPDF plugin's
   ``initialize()`` hook must check for the package actually installed
   (``paddlex``), not a legacy package name (``paddleocr``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Private-module detector ──────────────────────────────────────────────


def _find_private_paddlex_imports(source: str, filename: str) -> list[tuple[int, str]]:
    """Return a list of (lineno, import_line) for imports touching paddlex._*.

    Catches both forms::

        import paddlex._utils.logging
        from paddlex._internal import something
    """
    violations: list[tuple[int, str]] = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return violations

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # e.g. "import paddlex._utils" or "import paddlex._utils.logging"
                if _is_private_paddlex(alias.name):
                    violations.append((node.lineno, alias.name))

        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and _is_private_paddlex(node.module)
        ):
            names = ", ".join(a.name for a in node.names)
            violations.append((node.lineno, f"{node.module} ({names})"))

    return violations


def _is_private_paddlex(module_path: str) -> bool:
    """Return True if *module_path* references a private (``_*``) PaddleX
    submodule.

    Matches ``paddlex._anything`` or ``paddlex.something._anything``.
    Does NOT flag top-level ``paddlex`` or public subpackages like
    ``paddlex.utils`` (without underscore prefix).
    """
    parts = module_path.split(".")
    if parts[0] != "paddlex":
        return False
    # Any component after "paddlex" that starts with '_' is private
    return any(part.startswith("_") for part in parts[1:])


@pytest.mark.parametrize(
    "src_file",
    [
        REPO_ROOT / "paddlex_helpers.py",
        REPO_ROOT / "server.py",
        REPO_ROOT / "worker.py",
    ],
)
def test_no_private_paddlex_imports(src_file: Path) -> None:
    """Source files must not import from paddlex._* private modules."""
    source = src_file.read_text(encoding="utf-8")
    violations = _find_private_paddlex_imports(source, str(src_file))

    assert not violations, (
        f"Private PaddleX import(s) in {src_file.name}:\n"
        + "\n".join(f"  line {ln}: {mod}" for ln, mod in violations)
        + "\nUse logging.getLogger('paddlex') instead of importing paddlex._utils.*"
    )


# ── OCRmyPDF plugin dependency check ─────────────────────────────────────


def test_plugin_initialize_checks_for_paddlex_not_paddleocr() -> None:
    """The OCRmyPDF plugin must check for `paddlex`, not `paddleocr`."""
    init_file = REPO_ROOT / "ocrmypdf_paddleocr" / "__init__.py"
    source = init_file.read_text(encoding="utf-8")

    # The initialize() hook should try to import paddlex, not paddleocr
    # We look for the import inside a try/except block.
    tree = ast.parse(source, filename=str(init_file))

    found_paddlex_import = False
    found_paddleocr_import = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "paddlex":
                    found_paddlex_import = True
                if alias.name == "paddleocr":
                    found_paddleocr_import = True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "paddlex":
                found_paddlex_import = True
            if node.module == "paddleocr":
                found_paddleocr_import = True

    assert not found_paddleocr_import, (
        "ocrmypdf_paddleocr/__init__.py still imports 'paddleocr'. "
        "The project uses 'paddlex' — update the dependency check."
    )
    assert found_paddlex_import, (
        "ocrmypdf_paddleocr/__init__.py should import 'paddlex' to verify "
        "the dependency is available."
    )
