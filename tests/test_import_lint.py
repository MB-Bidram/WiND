"""Tests for the WiND / WiNC architectural layering rules.

These tests enforce the dependency direction: WiNC <- WiND.
WiNC must never import from WiND, and WiND must never import
FlashPKM directly (all PKM access goes through winc.pkm).
"""

import ast
import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modules under wind/ must not import from flashpkm directly.
WIND_PKGS = ["wind"]
WINC_PKGS = ["winc"]


def _collect_python_files(pkg_dirs: list[str]) -> list[Path]:
    files = []
    for pkg_dir in pkg_dirs:
        pkg_path = PROJECT_ROOT / pkg_dir
        if pkg_path.exists():
            for f in pkg_path.rglob("*.py"):
                files.append(f)
    return files


def _get_imports(source: str) -> list[tuple[str, str]]:
    """Extract (module, import_statement) tuples from source code."""
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.append((node.module, node.lineno))
    return imports


class TestArchitectureLayering:
    """Verify the WiNC <- WiND dependency direction."""

    def test_wind_does_not_import_flashpkm_directly(self):
        """WiND must not import flashpkm directly — use winc.pkm instead."""
        wind_files = _collect_python_files(WIND_PKGS)
        violations = []
        for f in wind_files:
            source = f.read_text(encoding="utf-8")
            for module, lineno in _get_imports(source):
                if module == "flashpkm" or module.startswith("flashpkm."):
                    violations.append(f"{f}:{lineno}: imports flashpkm directly")
        assert not violations, "Direct flashpkm imports in wind/:\n" + "\n".join(violations)

    def test_winc_does_not_import_wind(self):
        """WiNC must never import wind (the frontend)."""
        winc_files = _collect_python_files(WINC_PKGS)
        violations = []
        for f in winc_files:
            source = f.read_text(encoding="utf-8")
            for module, lineno in _get_imports(source):
                if module == "wind" or module.startswith("wind."):
                    violations.append(f"{f}:{lineno}: imports wind")
        assert not violations, "wind imports in winc/:\n" + "\n".join(violations)

    def test_wind_imports_go_through_winc(self):
        """WiND must access low-level modules through winc, not internal paths."""
        wind_files = _collect_python_files(WIND_PKGS)
        violations = []
        for f in wind_files:
            source = f.read_text(encoding="utf-8")
            for module, lineno in _get_imports(source):
                # wind.language.model should import from winc.architecture, not wind.architecture
                if module.startswith("wind.engine") or module.startswith("wind._internal"):
                    # Allow wind.language and wind.api internal relative imports
                    rel = f.relative_to(PROJECT_ROOT)
                    if not str(rel).startswith("wind/language/") and not str(rel).startswith("wind/api/"):
                        violations.append(f"{f}:{lineno}: imports {module}")
                    elif module.startswith("wind.engine"):
                        violations.append(f"{f}:{lineno}: imports {module} (should be winc.*)")
        assert not violations, "Improper wind imports:\n" + "\n".join(violations)

    def test_import_wind_succeeds(self):
        """Importing wind should not raise."""
        importlib.invalidate_caches()
        importlib.import_module("wind")
        import wind
        assert wind.__version__ == "2.1.0"

    def test_import_winc_succeeds(self):
        """Importing winc should not raise."""
        importlib.invalidate_caches()
        importlib.import_module("winc")
        import winc
        assert hasattr(winc, "WindModule")
