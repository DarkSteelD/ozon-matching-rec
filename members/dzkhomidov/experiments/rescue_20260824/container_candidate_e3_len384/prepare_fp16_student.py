from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).parent
SOURCE = Path("/home/dzkhomidov/matching-work/rescue_20260824/positive_composition/final_all_e3_len384/model")
DEST = ROOT / "package" / "models" / "student"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    assert (SOURCE / "model.safetensors").is_file()
    DEST.mkdir(parents=True, exist_ok=True)
    source_model = AutoModelForSequenceClassification.from_pretrained(
        SOURCE, local_files_only=True).half().eval()
    source_model.save_pretrained(DEST)
    # Preserve exact tokenizer/config bytes from the durable FP32 checkpoint.
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(SOURCE / name, DEST / name)

    package_model = AutoModelForSequenceClassification.from_pretrained(
        DEST, local_files_only=True).half().eval()
    source_state = source_model.state_dict()
    package_state = package_model.state_dict()
    assert source_state.keys() == package_state.keys()
    mismatches = []
    for name in source_state:
        if not torch.equal(source_state[name], package_state[name]):
            mismatches.append(name)
    assert not mismatches, mismatches[:10]

    tokenizer = AutoTokenizer.from_pretrained(DEST, local_files_only=True)
    encoded = tokenizer(
        ["товар alpha | Категория | бренд:x", "кроссовки красные | Обувь | бренд:y"],
        ["товар alpha | Категория | бренд:x", "кроссовки синие | Обувь | бренд:y"],
        truncation=True, max_length=384, padding=True, return_tensors="pt")
    use_tt = getattr(source_model.config, "type_vocab_size", 0) > 1
    if not use_tt:
        encoded.pop("token_type_ids", None)
    # CPU float32 math from the same quantized-half weights is deterministic
    # and verifies the serialized model path, class, config and tokenizer.
    with torch.inference_mode():
        logits_source = source_model.float()(**encoded).logits
        logits_package = package_model.float()(**encoded).logits
    max_abs = float((logits_source - logits_package).abs().max())
    prediction_bit_equal = bool(torch.equal(logits_source, logits_package))
    assert prediction_bit_equal and max_abs == 0.0

    report = {
        "source_fp32_model_bytes": (SOURCE / "model.safetensors").stat().st_size,
        "source_fp32_model_sha256": sha256(SOURCE / "model.safetensors"),
        "package_fp16_model_bytes": (DEST / "model.safetensors").stat().st_size,
        "package_fp16_model_sha256": sha256(DEST / "model.safetensors"),
        "state_tensor_count": len(source_state),
        "state_tensor_mismatches_after_source_half_vs_saved_half": len(mismatches),
        "config_source_sha256": sha256(SOURCE / "config.json"),
        "config_package_sha256": sha256(DEST / "config.json"),
        "tokenizer_source_sha256": sha256(SOURCE / "tokenizer.json"),
        "tokenizer_package_sha256": sha256(DEST / "tokenizer.json"),
        "sample_prediction_bit_equal": prediction_bit_equal,
        "sample_prediction_max_abs": max_abs,
    }
    (ROOT / "FP16_CONVERSION.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
