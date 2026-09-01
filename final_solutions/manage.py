#!/usr/bin/env python3
"""Fetch, verify, extract, and reproducibly repack the two ODS final solutions.

The script intentionally uses only the Python standard library.  The immutable
archive/member hashes live in ``manifest.json`` next to this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parent / "artifacts" / "final_solutions"
CHUNK = 4 * 1024 * 1024


def load_manifest() -> dict:
    return json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))


def digest_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(CHUNK):
            result.update(block)
    return result.hexdigest()


def selected_solutions(args: argparse.Namespace, manifest: dict) -> list[str]:
    names = args.solution or list(manifest["solutions"])
    unknown = sorted(set(names) - set(manifest["solutions"]))
    if unknown:
        raise SystemExit(f"unknown solution(s): {', '.join(unknown)}")
    return names


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"unsafe ZIP member: {name!r}")
    return member


def expected_members(spec: dict) -> dict[str, dict]:
    return {entry["path"]: entry for entry in spec["members"]}


def verify_archive(path: Path, spec: dict, *, verbose: bool = True) -> None:
    archive = spec["archive"]
    actual_size = path.stat().st_size
    if actual_size != archive["size"]:
        raise ValueError(
            f"{path}: size {actual_size}, expected {archive['size']}"
        )
    actual_hash = digest_file(path)
    if actual_hash != archive["sha256"]:
        raise ValueError(
            f"{path}: sha256 {actual_hash}, expected {archive['sha256']}"
        )

    expected = expected_members(spec)
    with zipfile.ZipFile(path) as bundle:
        all_infos = bundle.infolist()
        infos = [info for info in all_infos if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"{path}: duplicate ZIP member names")
        for info in all_infos:
            safe_member(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"{path}: symbolic links are not allowed: {info.filename}")
        if set(names) != set(expected):
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise ValueError(f"{path}: member mismatch; missing={missing}, extra={extra}")
        for info in infos:
            wanted = expected[info.filename]
            digest = hashlib.sha256()
            with bundle.open(info) as source:
                while block := source.read(CHUNK):
                    digest.update(block)
            if info.file_size != wanted["size"] or digest.hexdigest() != wanted["sha256"]:
                raise ValueError(f"{path}: content mismatch: {info.filename}")
    if verbose:
        print(f"OK  {path.name}  sha256={actual_hash}")


def download(url: str, target: Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    if start > expected_size:
        raise ValueError(f"partial file is larger than expected: {partial}")
    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        resumed = start and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if start and not resumed:
            start = 0
        received = start
        with partial.open(mode) as output:
            while block := response.read(CHUNK):
                output.write(block)
                received += len(block)
                print(
                    f"\r{target.name}: {received:,}/{expected_size:,} bytes",
                    end="",
                    flush=True,
                )
    print()
    if partial.stat().st_size != expected_size:
        raise ValueError(f"incomplete download retained at {partial}")
    os.replace(partial, target)


def command_fetch(args: argparse.Namespace, manifest: dict) -> None:
    output = args.output.resolve()
    for name in selected_solutions(args, manifest):
        spec = manifest["solutions"][name]
        archive = spec["archive"]
        target = output / archive["filename"]
        if not target.exists():
            download(archive["url"], target, archive["size"])
        verify_archive(target, spec)


def command_verify(args: argparse.Namespace, manifest: dict) -> None:
    output = args.output.resolve()
    for name in selected_solutions(args, manifest):
        spec = manifest["solutions"][name]
        verify_archive(output / spec["archive"]["filename"], spec)


def command_extract(args: argparse.Namespace, manifest: dict) -> None:
    output = args.output.resolve()
    for name in selected_solutions(args, manifest):
        spec = manifest["solutions"][name]
        archive = output / spec["archive"]["filename"]
        verify_archive(archive, spec, verbose=False)
        target = output / name
        if target.exists():
            raise FileExistsError(
                f"refusing to overwrite {target}; move it aside and rerun"
            )
        output.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=output))
        try:
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    member = safe_member(info.filename)
                    destination = temporary.joinpath(*member.parts)
                    if info.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, destination.open("wb") as sink:
                        shutil.copyfileobj(source, sink, CHUNK)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        print(f"OK  extracted {name} -> {target}")


def command_pack(args: argparse.Namespace, manifest: dict) -> None:
    name = args.solution
    spec = manifest["solutions"][name]
    source = args.source.resolve()
    expected = expected_members(spec)
    present = {
        path.relative_to(source).as_posix(): path
        for path in source.rglob("*")
        if path.is_file()
    }
    if set(present) != set(expected):
        missing = sorted(set(expected) - set(present))
        extra = sorted(set(present) - set(expected))
        raise ValueError(f"source layout mismatch; missing={missing}, extra={extra}")
    if not args.allow_content_drift:
        for relative, path in present.items():
            wanted = expected[relative]
            if path.stat().st_size != wanted["size"] or digest_file(path) != wanted["sha256"]:
                raise ValueError(
                    f"{relative} differs from the submitted final; "
                    "pass --allow-content-drift only for an intentional retrain"
                )

    target = args.archive.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for relative in sorted(present):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with present[relative].open("rb") as source_file, bundle.open(info, "w") as sink:
                shutil.copyfileobj(source_file, sink, CHUNK)
    os.replace(temporary, target)
    print(f"OK  packed {target}  size={target.stat().st_size} sha256={digest_file(target)}")


def build_parser(manifest: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    choices = list(manifest["solutions"])
    for command in ("fetch", "verify", "extract"):
        child = subparsers.add_parser(command)
        child.add_argument("--solution", action="append", choices=choices)
        child.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--solution", required=True, choices=choices)
    pack.add_argument("--source", required=True, type=Path)
    pack.add_argument("--archive", required=True, type=Path)
    pack.add_argument("--allow-content-drift", action="store_true")
    return parser


def main() -> None:
    manifest = load_manifest()
    args = build_parser(manifest).parse_args()
    {
        "fetch": command_fetch,
        "verify": command_verify,
        "extract": command_extract,
        "pack": command_pack,
    }[args.command](args, manifest)


if __name__ == "__main__":
    main()
