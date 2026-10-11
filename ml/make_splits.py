"""Patient-level train/test split for the AI segmenter (stdlib only).

Reads
  ml/test_patients_private.json  {ReMIND ID: sequence the tumor SEG references} (gitignored)
Writes
  ml/splits.json          anonymous IDs only (safe to commit)
  ml/splits_private.json  anon ID -> ReMIND ID (gitignored, never commit)

Test patients are hand-picked: all have a tumor, cerebrum and ventricles SEG plus an axial T1-post,
so every layer can be scored, and they are balanced by the sequence the tumor was drawn on
(3 on T1-post, 3 on T2). The classical segmenter is scored on the same 6.
Real IDs live only in the gitignored files, because the repo is public.
Anonymous IDs come from a seeded shuffle, so they don't reveal the ReMIND number but are reproducible.

Usage: python -I ml/make_splits.py --remind data/archive/remind
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20261010


def labels_of(patient: Path) -> dict:
    names = [d.name for d in (patient / "Preop").iterdir() if d.is_dir() and (d / ".complete").exists()]
    return {
        "tumor": any("_SEG_tumor_seg" in n for n in names),
        "cerebrum": any("_SEG_cerebrum" in n for n in names),
        "ventricles": any("_SEG_ventricles" in n for n in names),
        "t1_post_axial": any("_MR_3D_AX_T1_postcontrast" in n for n in names),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remind", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    TEST = json.loads((args.out / "test_patients_private.json").read_text(encoding="utf-8"))

    patients = sorted(p for p in args.remind.glob("ReMIND-*") if (p / "Preop").is_dir())
    info = {p.name: labels_of(p) for p in patients}
    usable = [pid for pid, lab in info.items() if lab["tumor"]]  # no tumor label -> not used yet

    missing = [pid for pid in TEST if pid not in usable or not all(info[pid].values())]
    if missing:
        raise SystemExit(f"test patients lack labels or data: {len(missing)}")

    shuffled = usable[:]
    random.Random(SEED).shuffle(shuffled)
    anon = {pid: f"rm{i + 1:03d}" for i, pid in enumerate(shuffled)}

    def entry(pid):
        return {"id": anon[pid], **info[pid]}

    public = {
        "seed": SEED,
        "note": "Patient-level split; no patient appears in both. Anonymous IDs only.",
        "test": sorted((entry(p) | {"tumor_drawn_on": TEST[p]} for p in TEST), key=lambda e: e["id"]),
        "train": sorted((entry(p) for p in usable if p not in TEST), key=lambda e: e["id"]),
        "unused": len(patients) - len(usable),
    }
    (args.out / "splits.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    (args.out / "splits_private.json").write_text(
        json.dumps({anon[p]: p for p in usable}, indent=2) + "\n", encoding="utf-8")
    print(f"train {len(public['train'])}  test {len(public['test'])}  unused {public['unused']}")


if __name__ == "__main__":
    main()
