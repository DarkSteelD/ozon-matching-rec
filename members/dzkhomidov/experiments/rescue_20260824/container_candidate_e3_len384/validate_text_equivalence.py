from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
ITEMS = Path("/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/data/raw/items.parquet")
MATCHES = Path("/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/data/raw/matches.parquet")
spec = importlib.util.spec_from_file_location("run_v5", ROOT / "solo" / "run_v5.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def payload(items, matches):
    id2text = run.build_texts(items)
    id2cat = dict(zip(items["id"], items["category"]))
    id2attr = dict(zip(items["id"], items["attributes"]))
    t1 = [id2text.get(i, "") for i in matches["id1"]]
    t2 = [id2text.get(i, "") for i in matches["id2"]]
    p1, p2, flags, cache = [], [], [], {}
    for i1, i2 in zip(matches["id1"], matches["id2"]):
        for iid in (i1, i2):
            if iid not in cache:
                name = id2text.get(iid, "").split(" | ")[0]
                category = id2cat.get(iid) or ""
                fashion = category in run.FASHION
                kv = run.parse_kv(run.compact_attrs(id2attr.get(iid)))
                cache[iid] = (name, category, kv,
                              run.prio_attrs(run.compact_attrs(id2attr.get(iid)), fashion))
        n1, c1, kv1, a1 = cache[i1]
        n2, c2, kv2, a2 = cache[i2]
        diff = (" @@ сравнение: цвет=" + run.cmp_tok(run.getv(kv1, ["цвет"]), run.getv(kv2, ["цвет"]))
                + "; артикул=" + run.cmp_tok(
                    run.getv(kv1, ["артикул", "модель", "код товара", "партномер"]),
                    run.getv(kv2, ["артикул", "модель", "код товара", "партномер"])))
        p1.append(f"{n1} | {c1} | {a1}{diff}")
        p2.append(f"{n2} | {c2} | {a2}{diff}")
        flags.append(c1 in run.FASHION and run.size_mismatch(run.size_sets(kv1), run.size_sets(kv2)))
    return t1, t2, p1, p2, np.asarray(flags, dtype=np.uint8)


def hash_strings(values):
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def main():
    started = time.time()
    matches = pd.read_parquet(MATCHES)[["id1", "id2"]]
    needed = np.unique(np.concatenate((matches["id1"].to_numpy(), matches["id2"].to_numpy())))
    full_items = pd.read_parquet(ITEMS, columns=["id", "name", "category", "attributes"])
    old = payload(full_items, matches)
    del full_items
    filtered_items = run.read_needed_items(ITEMS, needed)
    new = payload(filtered_items, matches)
    names = ("t1", "t2", "p1", "p2")
    mismatches = {name: int(sum(a != b for a, b in zip(old[i], new[i])))
                  for i, name in enumerate(names)}
    mismatches["penalty_flags"] = int(np.count_nonzero(old[4] != new[4]))
    hashes = {"old_" + name: hash_strings(old[i]) for i, name in enumerate(names)}
    hashes.update({"new_" + name: hash_strings(new[i]) for i, name in enumerate(names)})
    hashes["old_penalty_flags"] = hashlib.sha256(old[4].tobytes()).hexdigest()
    hashes["new_penalty_flags"] = hashlib.sha256(new[4].tobytes()).hexdigest()
    result = {"pairs": len(matches), "needed_ids": len(needed),
              "retained_items": len(filtered_items), "mismatches": mismatches,
              "hashes": hashes, "runtime_seconds": time.time() - started}
    assert not any(mismatches.values()), result
    (ROOT / "text_equivalence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
