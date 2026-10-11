"""Train the AI segmenter: nnU-Net v2, 3D full resolution, on all training cases (fold "all").

Steps: plan + preprocess (once) -> train -> copy the final model to ml/weights/cortexview_nnunet/
(gitignored), where backend/cortexview/segment_ai loads it. Held-out test cases (imagesTs) are never used.

The trainer is built here rather than through nnU-Net's command line, so that:
  - pinned-memory batch copies are off (on Windows they fail with "CUDA error: resource already mapped"),
  - the epoch count can be anything,
  - nnU-Net's end-of-training validation is skipped: with fold "all" it would only re-predict the
    training scans; ml/evaluate.py scores the held-out test scans instead.
torch.compile needs Triton, which Windows lacks, so it's off. nnU-Net folders live in ml/nnunet/
(gitignored: patient-derived volumes).

Usage (repo root, AI venv):
  backend/.venv-ai/Scripts/python.exe -I ml/train.py --epochs 250
  backend/.venv-ai/Scripts/python.exe -I ml/train.py --epochs 250 --continue     # resume after a stop
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent                           # venv Scripts folder
WEIGHTS = ROOT / "ml/weights/cortexview_nnunet"


def setup_env(nn: Path, n_proc: int) -> None:
    os.environ.update({"nnUNet_raw": str(nn / "raw"), "nnUNet_preprocessed": str(nn / "preprocessed"),
                       "nnUNet_results": str(nn / "results"), "nnUNet_compile": "f",
                       "nnUNet_n_proc_DA": str(n_proc), "PYTHONUNBUFFERED": "1"})


def sh(*args):
    print(">", " ".join(map(str, args)), flush=True)
    subprocess.run([str(a) for a in args], env=os.environ, check=True)


def make_trainer(epochs: int, iters: int):
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

    class NoPinAugmenter(NonDetMultiThreadedAugmenter):
        def __init__(self, *a, **k):
            k["pin_memory"] = False
            super().__init__(*a, **k)
    base.NonDetMultiThreadedAugmenter = NoPinAugmenter      # used by nnUNetTrainer.get_dataloaders

    class CortexViewTrainer(base.nnUNetTrainer):
        # same signature as nnUNetTrainer: it records its init arguments by name for checkpoints
        def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict, device=None):
            import torch
            super().__init__(plans, configuration, fold, dataset_json, device or torch.device("cuda"))
            self.num_epochs = epochs
            if iters:
                self.num_iterations_per_epoch = iters
    return CortexViewTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--continue", dest="cont", action="store_true")
    ap.add_argument("--gpu-gb", type=float, default=7.0, help="GPU memory the plan may use")
    ap.add_argument("--workers", type=int, default=4, help="data-augmentation processes (RAM-bound)")
    ap.add_argument("--dataset", default="Dataset501_CortexView")
    ap.add_argument("--nnunet-dir", type=Path, default=ROOT / "ml/nnunet")
    ap.add_argument("--iters", type=int, default=0, help="iterations per epoch (0 = nnU-Net's 250); for tests")
    ap.add_argument("--no-copy", action="store_true", help="don't copy the model to ml/weights")
    a = ap.parse_args()
    setup_env(a.nnunet_dir, a.workers)
    did = int(a.dataset[len("Dataset"):len("Dataset") + 3])

    pre = a.nnunet_dir / "preprocessed" / a.dataset
    if not (pre / "nnUNetPlans_3d_fullres").exists():
        sh(BIN / "nnUNetv2_extract_fingerprint.exe", "-d", did, "--verify_dataset_integrity", "-np", 4)
        sh(BIN / "nnUNetv2_plan_experiment.exe", "-d", did, "-gpu_memory_target", a.gpu_gb)
        sh(BIN / "nnUNetv2_preprocess.exe", "-d", did, "-c", "3d_fullres", "-np", 4)

    import torch
    from batchgenerators.utilities.file_and_folder_operations import load_json
    plans = load_json(pre / "nnUNetPlans.json")
    plans["continue_training"] = a.cont                     # as nnU-Net's get_trainer_from_args does
    trainer = make_trainer(a.epochs, a.iters)(plans, "3d_fullres", "all", load_json(pre / "dataset.json"),
                                              torch.device("cuda"))
    if a.cont:
        for name in ("checkpoint_latest.pth", "checkpoint_final.pth"):
            ck = Path(trainer.output_folder) / name
            if ck.exists():
                trainer.load_checkpoint(str(ck))
                break
    torch.backends.cudnn.benchmark = True
    trainer.run_training()

    if a.no_copy:
        return
    run = Path(trainer.output_folder_base)
    if WEIGHTS.exists():
        shutil.rmtree(WEIGHTS)
    (WEIGHTS / "fold_all").mkdir(parents=True)
    for f in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        shutil.copy2(run / f, WEIGHTS / f)
    # the predictor looks the trainer class up by name inside nnunetv2; ours only changes the training
    # loop (same network), so the copy is labelled as nnU-Net's own trainer
    ck = torch.load(run / "fold_all" / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    ck["trainer_name"] = "nnUNetTrainer"
    torch.save(ck, WEIGHTS / "fold_all" / "checkpoint_final.pth")
    print(f"weights copied to {WEIGHTS}")


if __name__ == "__main__":
    main()
