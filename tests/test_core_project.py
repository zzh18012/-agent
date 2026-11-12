from pathlib import Path

from mmos.core import ProjectPaths, sha256_file

EXPECTED_HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    target.write_bytes(b"hello")
    assert sha256_file(target) == EXPECTED_HELLO_SHA256


def test_project_paths_are_derived_from_root(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path / "demo")
    assert paths.config_file == tmp_path / "demo" / "project.yaml"
    assert paths.manifest_file == tmp_path / "demo" / "manifests" / "project_manifest.json"
    assert paths.raw_data_dir == tmp_path / "demo" / "data" / "raw"
    assert paths.verified_dir == tmp_path / "demo" / "verified"
