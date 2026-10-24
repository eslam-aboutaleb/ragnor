"""Verify actual wheel/sdist license metadata and notice bytes against the source."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def check(artifact: Path, root: Path) -> None:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            files = {
                name: archive.read(name) for name in archive.namelist() if not name.endswith("/")
            }
        metadata = next(
            data for name, data in files.items() if name.endswith(".dist-info/METADATA")
        )
        prefix = next(
            name.rsplit("/", 1)[0] for name in files if name.endswith(".dist-info/METADATA")
        )
        notices = {notice: f"{prefix}/licenses/{notice}" for notice in ("LICENSE", "NOTICE")}
    else:
        with tarfile.open(artifact, "r:gz") as archive:
            files = {}
            for member in archive.getmembers():
                if member.isfile():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    files[member.name] = stream.read()
        prefix = next(iter(files)).split("/", 1)[0]
        metadata = files[f"{prefix}/PKG-INFO"]
        notices = {notice: f"{prefix}/{notice}" for notice in ("LICENSE", "NOTICE")}
    headers = BytesParser().parsebytes(metadata)
    if headers.get("License-Expression") != "Elastic-2.0":
        raise ValueError(f"{artifact}: wrong License-Expression")
    for notice, path in notices.items():
        if files.get(path) != (root / notice).read_bytes():
            raise ValueError(f"{artifact}: missing or modified {notice}")
        if notice not in headers.get_all("License-File", []):
            raise ValueError(f"{artifact}: missing License-File: {notice}")
    print(f"{artifact.name}: Elastic-2.0 metadata and exact LICENSE/NOTICE verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for artifact in args.artifacts:
        check(artifact, root)


if __name__ == "__main__":
    main()
