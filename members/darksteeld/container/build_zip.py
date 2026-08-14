"""Assemble a submission archive from a container directory.

Copies the shared feature module in as ``src/pair_features.py`` so the archive
is self-contained while the repository keeps one source of truth, checks the
contract files are present, and writes a zip into ``members/<member>/submissions/``
(gitignored — archives are build output, not source).

    .venv/bin/python members/darksteeld/container/build_zip.py lgbm_cheap

The archive is never uploaded from here. Submitting is a manual step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]  # members/darksteeld/container -> repo root
SHARED_SRC = REPOSITORY_ROOT / "members" / "darksteeld" / "src" / "pair_features.py"
SUBMISSIONS = REPOSITORY_ROOT / "members" / "darksteeld" / "submissions"

REQUIRED = ("metadata.json", "run.py")
EXCLUDE = {"build_artifact.py", "__pycache__", ".DS_Store"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", help="directory name under members/darksteeld/container/")
    parser.add_argument("--out-dir", type=Path, default=SUBMISSIONS)
    parser.add_argument("--shared", type=Path, default=SHARED_SRC,
                        help="module copied into src/ inside the archive; 'none' to skip")
    args = parser.parse_args()

    source = HERE / args.container
    if not source.is_dir():
        raise SystemExit(f"no such container directory: {source}")

    if str(args.shared).lower() != "none":
        if not args.shared.is_file():
            raise SystemExit(f"shared module not found: {args.shared}")
        (source / "src").mkdir(exist_ok=True)
        shutil.copy2(args.shared, source / "src" / args.shared.name)
        print(f"vendored {args.shared.name} -> {args.container}/src/")

    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"missing contract files: {', '.join(missing)}")

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    for key in ("image", "entry_point"):
        if not metadata.get(key):
            raise SystemExit(f"metadata.json lacks {key!r}")

    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and not any(part in EXCLUDE for part in path.parts)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / f"{args.container}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as sink:
        for path in files:
            sink.write(path, path.relative_to(source).as_posix())

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"\n{archive}  {archive.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 {digest}")
    print(f"image  {metadata['image']}")
    print(f"entry  {metadata['entry_point']}")
    print("contents:")
    for path in files:
        print(f"  {path.relative_to(source).as_posix():<34} {path.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
