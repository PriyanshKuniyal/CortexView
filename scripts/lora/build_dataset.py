"""
Step 4: turn your hand-picked candidates into an ai-toolkit edit-LoRA dataset.

picked.txt: one candidate filename per line (e.g. ReMIND-003_s62__s1.png), max one per guide.

    python build_dataset.py --picked picked.txt --guides guides --candidates candidates --out dataset

Produces:
    dataset/reference/<id>.png   guide (input)
    dataset/target/<id>.png      illustration (output)
    dataset/target/<id>.txt      edit instruction caption
"""
import argparse
import pathlib

from PIL import Image

# 6 phrasings with the same activation phrase, so the LoRA tolerates prompt wording
# (the BFL guide notes that only 2-3 phrasings generalize poorly).
CAPTIONS = [
    "turn this brain segmentation map into a CRTXV1EW illustration",
    "render this color-coded brain map as a CRTXV1EW illustration",
    "convert the labeled brain slice into a CRTXV1EW illustration",
    "redraw this segmentation as a CRTXV1EW illustration, keeping every region in place",
    "make a CRTXV1EW illustration from this brain map",
    "transform this brain label map into a CRTXV1EW illustration with the same layout",
]

ap = argparse.ArgumentParser()
ap.add_argument("--picked", default="picked.txt")
ap.add_argument("--guides", default="guides")
ap.add_argument("--candidates", default="candidates")
ap.add_argument("--out", default="dataset")
ap.add_argument("--size", type=int, default=1024)
args = ap.parse_args()

names = [l.strip() for l in pathlib.Path(args.picked).read_text().splitlines() if l.strip()]
stems = [n.split("__s")[0] for n in names]
assert len(set(stems)) == len(stems), "picked.txt has two samples from the same guide"

ref_dir = pathlib.Path(args.out) / "reference"
tgt_dir = pathlib.Path(args.out) / "target"
ref_dir.mkdir(parents=True, exist_ok=True)
tgt_dir.mkdir(parents=True, exist_ok=True)

for i, (name, stem) in enumerate(zip(names, stems)):
    pid = f"{i:03d}"
    guide = Image.open(pathlib.Path(args.guides) / f"{stem}.png").convert("RGB")
    guide.resize((args.size, args.size), Image.NEAREST).save(ref_dir / f"{pid}.png")
    target = Image.open(pathlib.Path(args.candidates) / name).convert("RGB")
    target.resize((args.size, args.size), Image.LANCZOS).save(tgt_dir / f"{pid}.png")
    (tgt_dir / f"{pid}.txt").write_text(CAPTIONS[i % len(CAPTIONS)])

print(f"wrote {len(names)} pairs to {args.out}/")
