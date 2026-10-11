# CortexView LoRA Fine-Tuning — Handoff for Claude Code

## Mission

Continue the CortexView project from its current state and prepare a fine-tune of **FLUX.2 [klein] base-9B** using an **edit LoRA** with **Ostris AI Toolkit** on **Lightning AI**. Work through the existing repository and scripts rather than rebuilding the pipeline from scratch.

The user edits the repository in Cursor and connects to the Lightning Studio over SSH. The Lightning GPU is currently **disabled to avoid spending credits**. Keep it disabled while inspecting files, fixing paths, transferring data, and preparing the environment. Only ask the user to enable it when all CPU-side checks are complete and training is ready to start.

## Project context

- Project: **CortexView**
- Repository: `https://github.com/PriyanshKuniyal/Plie`
- Local Windows repo: `C:\Users\simpl\OneDrive\Desktop\Plie`
- Lightning repo (confirmed): `/teamspace/studios/this_studio/Plie`
- AI Toolkit checkout was shown at `~/ai-toolkit` on Lightning.
- Base model is documented as already downloaded at:
  `/teamspace/studios/this_studio/flux2-klein-base-9b`
- Intended trainer: `ostris/ai-toolkit`
- Intended architecture: `flux2_klein_9b`
- Intended training mode: edit LoRA using `control_path`
- Existing training config: `scripts/lora/train_klein9b_edit.yaml`
- Existing scripts:
  - `scripts/lora/make_candidates.py`
  - `scripts/lora/fidelity_filter.py`
  - `scripts/lora/build_dataset.py`
- The project map records initial planned values of LoRA rank 32, learning rate `1e-4`, 2,000 steps, and checkpoint interval 250. **Verify the actual YAML and toolkit schema; do not assume these values or keys are currently correct.**

## Product/pipeline intent

CortexView takes a brain MRI slice (DICOM), uses deterministic segmentation/masks and math for anatomy/measurements, then uses FLUX.2 klein plus a project LoRA to render a color-coded educational illustration. The model should learn visual style; anatomy and numeric measurements must remain driven by masks and deterministic code. The app is an educational aid, not a diagnostic tool.

Frozen label palette in the project map:
- 0 background: RGB 0, 0, 0
- 1 skull/scalp: RGB 222, 184, 135
- 2 gray matter: RGB 196, 120, 170
- 3 white matter: RGB 245, 240, 225
- 4 CSF/ventricles: RGB 70, 140, 230
- 10+ findings/tumor: RGB 230, 40, 40

Palette source of truth is `scripts/synthseg/palette.json` and `shared/layers.json`. Do not alter label IDs or palette unless explicitly authorized.

## Work already completed

### On Windows
- ReMIND data work directory: `C:\Users\simpl\remind_work`
- SynthSeg ran for all 53 patients and finished with 0 failures.
- ReMIND SEG decoding bug was fixed in `scripts/synthseg/remind_synthseg.py`.
- Guides were built using the user's earlier choice **option B**: include review patients (`guides --include-review`).
- The report says the final output contains **80 guides: 53 tumor, 27 no-tumor**, from 53 patients, with a patient-level split of 40 train / 13 test patients.
- Guide files were copied into:
  `C:\Users\simpl\OneDrive\Desktop\Plie\data\guides\remind`
- That local folder was observed in File Explorer to contain:
  - `guide/`
  - `image/`
  - `labels/`
  - `meta/`
  - `guides_sheet.png` (PNG; shown as 706 KB)
  - `splits.csv` (shown as 1 KB)
- A report exists at:
  `C:\Users\simpl\OneDrive\Desktop\Plie\data\guides\OVERNIGHT_REPORT.md`
- Report location above and final guide generation are documented in the user's uploaded report. The report says the full guides were generated and copied, with the train/test split in `splits.csv`.
- The source guide directory was found to contain 240 `.png`, 80 `.json`, and 1 `.csv` files in total under `C:\Users\simpl\remind_work\guides`. The project-specific copied folder shown in Explorer has the four subfolders and two files listed above.
- A ZIP was successfully created locally:
  `C:\Users\simpl\Desktop\remind-guides.zip`
  Its reported size was 8,597,593 bytes (~8.6 MB). It was created using:
  `Compress-Archive -Path "C:\Users\simpl\OneDrive\Desktop\Plie\data\guides\remind\*" -DestinationPath "$HOME\Desktop\remind-guides.zip" -Force`

### On Lightning
- Remote repo confirmed at `/teamspace/studios/this_studio/Plie`.
- `~/ai-toolkit` exists and terminal prompt showed branch `main`.
- Base model directory is documented as:
  `/teamspace/studios/this_studio/flux2-klein-base-9b`
  Its actual contents have **not yet been verified in this conversation**.
- The remote destination directory was created:
  `/teamspace/studios/this_studio/Plie/data/guides/remind`
- Most recent checks showed that this remote destination is **empty**: only `.` and `..`; no `guide/`, `image/`, `labels/`, `meta/`, or `splits.csv`.
- Therefore, the guide ZIP has **not been successfully transferred or extracted on Lightning yet**.
- Direct Windows SSH test:
  `ssh -i "$HOME\.ssh\lightning_rsa" -o IdentitiesOnly=yes ssh.lightning.ai`
  failed with `simpl@ssh.lightning.ai: Permission denied (publickey)`.
- The SSH config shown by user contains:
  ```sshconfig
  Host ssh.lightning.ai
    IdentityFile C:\Users\simpl\.ssh\lightning_rsa
    IdentitiesOnly yes
    ServerAliveInterval 15
    ServerAliveCountMax 4
    StrictHostKeyChecking no
    UserKnownHostsFile=\\.\NUL
  ```
  No `User` line was shown. Cursor's existing Remote SSH session connects, but the exact working invocation/credentials have not been established.
- User tried uploading files into Cursor but remote verification still showed the directory empty. Do not assume upload succeeded.
- Terminal paste sometimes inserted `^[[200~` artifacts; this is a bracketed-paste issue, not a project problem. If it recurs, clear the command line and type the command normally or paste without leading escape characters.
- Lightning Studio blocks `python -m venv .venv` with:
  `Venv creation is not allowed. A Studio has a default conda environment (max 1 environment). Start a new Studio to create a new environment.`
  Use the existing managed Conda environment; do not attempt to create another venv. Do not install or upgrade dependencies blindly.

## Training dataset process still required

The project map's plan is:
1. Render about 3 candidates per training guide using the base model.
2. Filter candidates with fidelity metrics.
3. Have a human select 20–30 pairs.
4. Build paired dataset using `scripts/lora/build_dataset.py`, with trigger word `CRTXV1EW`.
5. Train the edit LoRA.
6. Select a checkpoint visually.
7. Evaluate base vs. LoRA on held-out test patients and report fidelity pass rate.

The dataset builder is described as creating `reference/` and `target/` folders with captions. Inspect scripts and `--help` to learn actual arguments. Do not invent CLI arguments. Check how the candidate-generation script loads the base model and whether the intended inference model/licensing access is available. The project map notes the separate distilled FLUX.2 klein 9B inference model's license still needed acceptance on Hugging Face; verify actual requirements before candidate generation.

The existing guides are *silver-standard* SynthSeg references, not ground truth. Keep that caveat in evaluation and any write-up. Do not claim clinical accuracy.

## Immediate tasks for Claude Code

Proceed in this order. Do not skip ahead.

### A. Inspect the remote repo and current state
On Lightning:
1. `cd /teamspace/studios/this_studio/Plie`
2. Inspect `git status --short`, `ls -lah data/guides`, and `ls -lah data/guides/remind`.
3. Inspect `scripts/lora/train_klein9b_edit.yaml` and all three `scripts/lora/*.py` scripts.
4. Read `data/guides/OVERNIGHT_REPORT.md` if it exists remotely.
5. Inspect `~/ai-toolkit` README and dependency/install instructions; check git revision and Python version.
6. Check the model directory contents and disk space without loading the model or enabling GPU.
7. Determine the exact, safe method to transfer `C:\Users\simpl\Desktop\remind-guides.zip` through the existing Cursor/Lightning connection. Do not invent SSH usernames or credentials. Do not ask the user to expose private keys/tokens. If necessary, give the user one specific GUI action or command based on the actual Remote-SSH log/config.
8. Once transferred, extract to `data/guides/remind/` and verify counts, files, and `splits.csv`. Do not overwrite valid files silently. Compare archive listing before extraction.

### B. Prepare the training setup without GPU
1. Verify all script and YAML paths.
2. Check whether `scripts/lora/train_klein9b_edit.yaml` expects the raw guides or a processed paired dataset. Confirm by reading `build_dataset.py` and config.
3. Run `--help` for scripts that support it; inspect argument parser source if not.
4. Check dataset counts and match files by IDs; ensure the train/test patient split is respected. Candidate generation/training must not use test patients.
5. Check `splits.csv` structure and map train/test guides accurately.
6. Verify AI Toolkit supports `flux2_klein_9b` and `control_path` in the installed revision. Do not use generic FLUX.1 assumptions.
7. Establish a compatible installation path using the Studio's existing Conda environment. Inspect first; do not blindly install dependencies or overwrite PyTorch.
8. Validate config keys, base-model path, output directory, checkpoint path, dataset path, captions, trigger word, and edit-conditioning paths.
9. Create or document a short smoke-test config separately from the full training config, preserving the full YAML.
10. Check model license and Hugging Face access status. Base model is gated and described as non-commercial; do not advise violating terms.

### C. Ask user to enable GPU only after readiness
Before asking user to enable GPU, report exactly what has passed and what remains. Then:
1. User enables GPU.
2. Verify `nvidia-smi`, CUDA/PyTorch availability, and GPU model/memory.
3. Run a short smoke test.
4. Only if successful, run the intended training job.
5. Use persistent output/checkpoint storage; spot GPU interruption risk matters.
6. Monitor memory and logs. Do not silently start a 2,000-step job or spend credits without making it clear to the user.
7. After training, evaluate selected checkpoint on held-out patients, compare base vs LoRA, report fidelity pass rate and limitations.

## Guardrails
- Keep GPU disabled until CPU-side setup, transfer, dataset verification, and config validation are complete.
- Never assume files uploaded successfully; verify remotely.
- Do not re-run SynthSeg or regenerate guides; those stages are complete on Windows.
- Do not upload DICOM/NIfTI source medical data to Lightning; transfer only the generated guide package unless a script demonstrably needs more.
- Do not expose or request private SSH keys, access tokens, or credentials.
- Do not create a new venv on Lightning.
- Do not commit/push without the user's explicit authorization.
- Do not change palette, labels, train/test split, or project scope without approval.
- Don't claim training has started or succeeded unless logs prove it.
- Keep the user informed with concrete, short steps and exact commands.
