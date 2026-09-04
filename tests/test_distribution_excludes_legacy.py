"""Assert built distributions exclude legacy sources and Enterprise imports."""

from __future__ import annotations

import ast
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("build")

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH_MARKERS = ("metapulsar/legacy/", "src/metapulsar/legacy/")


def _member_paths(archive_path: Path) -> list[str]:
    if archive_path.suffix == ".whl":
        with zipfile.ZipFile(archive_path) as zf:
            return zf.namelist()
    if archive_path.name.endswith(".tar.gz") or archive_path.suffix == ".tar":
        with tarfile.open(archive_path, "r:*") as tf:
            return [member.name for member in tf.getmembers()]
    raise AssertionError(f"Unexpected distribution artifact: {archive_path}")


def _contains_legacy_path(paths: list[str]) -> list[str]:
    hits = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if any(marker in normalized for marker in LEGACY_PATH_MARKERS):
            hits.append(path)
    return hits


def _read_archive_text(archive_path: Path, member: str) -> str:
    if archive_path.suffix == ".whl":
        with zipfile.ZipFile(archive_path) as zf:
            return zf.read(member).decode("utf-8")
    with tarfile.open(archive_path, "r:*") as tf:
        handle = tf.extractfile(member)
        assert handle is not None
        return handle.read().decode("utf-8")


def _production_py_members(archive_path: Path, paths: list[str]) -> list[str]:
    members = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if not normalized.endswith(".py"):
            continue
        if archive_path.suffix == ".whl":
            if normalized.startswith("metapulsar/") and "/legacy/" not in normalized:
                members.append(path)
        else:
            # sdist: production means paths under src/metapulsar/
            if "/src/metapulsar/" in f"/{normalized}" and "/legacy/" not in normalized:
                members.append(path)
    return members


def _enterprise_import_hits(source: str) -> list[str]:
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top == "enterprise":
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            top = node.module.split(".", 1)[0]
            if top == "enterprise":
                hits.append(node.module)
    return hits


@pytest.mark.slow
def test_wheel_and_sdist_exclude_legacy_and_enterprise_imports(tmp_path: Path) -> None:
    outdir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(outdir)],
        cwd=REPO_ROOT,
        check=True,
    )

    artifacts = sorted(outdir.iterdir())
    wheels = [p for p in artifacts if p.suffix == ".whl"]
    sdists = [p for p in artifacts if p.name.endswith(".tar.gz")]
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    assert len(sdists) == 1, f"expected one sdist, found {sdists}"

    for artifact in (*wheels, *sdists):
        paths = _member_paths(artifact)
        hits = _contains_legacy_path(paths)
        assert hits == [], f"{artifact.name} contains legacy paths: {hits}"

        for member in _production_py_members(artifact, paths):
            source = _read_archive_text(artifact, member)
            enterprise_hits = _enterprise_import_hits(source)
            assert (
                enterprise_hits == []
            ), f"{artifact.name}:{member} imports enterprise: {enterprise_hits}"
