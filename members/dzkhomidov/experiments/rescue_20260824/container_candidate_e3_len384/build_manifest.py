from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent
PACKAGE = ROOT / "package"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    metadata = json.loads((PACKAGE / "metadata.json").read_text())
    models = json.loads((PACKAGE / "models.json").read_text())
    assert metadata == {"image": "odsai/ecup26-matching-baseline:1.0",
                        "entry_point": "python -u run.py"}
    assert models[0] == {"path": "models/student", "max_len": 384,
                         "weight": 1.0, "texts": "prio"}
    assert len(models) == 1
    required = ["run.py", "metadata.json", "models.json",
                "models/student/config.json", "models/student/model.safetensors",
                "models/student/tokenizer.json", "models/student/tokenizer_config.json"]
    for relative in required:
        assert (PACKAGE / relative).is_file(), relative
    files = {}
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(PACKAGE))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    manifest = {
        "base_zip": "/home/dzkhomidov/matching-work/container/ecup_matching_consolidated_v2.zip",
        "base_zip_sha256": "7c115e7dd77653b4d19ecef80cf6772202e2e79fc690005a3970d0a68bc48bad",
        "metadata": metadata,
        "models": models,
        "files": files,
    }
    (ROOT / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
