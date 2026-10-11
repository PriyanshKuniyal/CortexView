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
  backend/.venv-ai/Scripts/python.exe -I ml/train.py --epochs 150
  backend/.venv-ai/Scripts/python.exe -I ml/train.py --epochs 150 --continue     # resume after a stop
(the epoch count may change on resume; the learning-rate schedule follows the new total)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(sys.executable).parent                           # venv Scripts folder
WEIGHTS = ROOT / "ml/weights/cortexview_nnunet"
EVERY = 10              # every 10 epochs: resumable checkpoint + usable model in WEIGHTS


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
            self.save_every = EVERY                         # checkpoint_latest.pth, for --continue
            if iters:
                self.num_iterations_per_epoch = iters
            self._it, self._t0 = 0, time.perf_counter()

        def on_train_epoch_start(self):
            super().on_train_epoch_start()
            self._it, self._t0 = 0, time.perf_counter()

        def train_step(self, batch):
            out = super().train_step(batch)
            self._it += 1
            n = self.num_iterations_per_epoch
            if self._it % 10 == 0 or self._it == n:         # live progress inside the epoch
                dt = time.perf_counter() - self._t0
                print(f"  epoch {self.current_epoch + 1}/{self.num_epochs}  step {self._it}/{n}  "
                      f"loss {float(out['loss']):.4f}  {dt / self._it:.2f} s/step  "
                      f"epoch ETA {dt / self._it * (n - self._it):.0f} s", flush=True)
            return out

        def on_epoch_end(self):
            super().on_epoch_end()                          # counts the epoch as done
            done = self.current_epoch
            left = (self.num_epochs - done) * (time.perf_counter() - self._t0)
            print(f"== epoch {done}/{self.num_epochs} done; about {left / 3600:.1f} h left", flush=True)
            if done % EVERY == 0 and done < self.num_epochs:
                export(self, f"epoch {done}")
    return CortexViewTrainer


def export(trainer, label: str) -> None:
    """Write the current model as a loadable nnU-Net folder in ml/weights/cortexview_nnunet/ (overwritten)."""
    import torch
    run = Path(trainer.output_folder_base)
    tmp = Path(trainer.output_folder) / "checkpoint_export.pth"
    trainer.save_checkpoint(str(tmp))
    if WEIGHTS.exists():
        shutil.rmtree(WEIGHTS)
    (WEIGHTS / "fold_all").mkdir(parents=True)
    for f in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        shutil.copy2(run / f, WEIGHTS / f)
    # the predictor looks the trainer class up by name inside nnunetv2; ours only changes the training
    # loop (same network), so the copy is labelled as nnU-Net's own trainer
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    ck["trainer_name"] = "nnUNetTrainer"
    torch.save(ck, WEIGHTS / "fold_all" / "checkpoint_final.pth")
    tmp.unlink()
    print(f"== model ({label}) saved to {WEIGHTS}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
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
        for name in ("checkpoint_latest.pth", "checkpoint_final.pth", "checkpoint_best.pth"):
            ck = Path(trainer.output_folder) / name
            if ck.exists():
                trainer.load_checkpoint(str(ck))
                print(f"resuming from {name} at epoch {trainer.current_epoch}", flush=True)
                break
    torch.backends.cudnn.benchmark = True
    trainer.run_training()

    if not a.no_copy:
        export(trainer, "final")


if __name__ == "__main__":
    main()
