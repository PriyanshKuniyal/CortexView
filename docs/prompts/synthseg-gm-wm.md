# Prompt: SynthSeg gray-matter, white-matter and outer-CSF reference labels for ReMIND

Paste everything below the line into a fresh Claude Code session opened at the repo root (`Plie/`).

<!-- CHANGE NOTE (2026-10-10, PK): scope widened from GM/WM only to GM/WM + outer CSF (SynthSeg label 24).
     ReMIND only labels the ventricles, so the CSF in the brain's folds and around the brain had no reference.
     The outputs are now also used as TRAINING labels for the AI segmenter (docs/contracts.md C6), not only for validation.
     The filename is kept so existing links still work. If you started from the older version: also write csf_outer.nii.gz
     and the new manifest columns (Steps 5–7). Everything else is unchanged. -->

---

You're helping build **CortexView**, a hackathon app. It splits a brain MRI slice into 5 layers (skull/scalp, gray matter, white matter, CSF, suspected lesion). Then it shows the layers as an illustration with metrics pins.
There are two interchangeable segmenters: a **classical, non-AI** one, and an **AI one** being trained on branch `segmentation-AI-version`.
Read `docs/contracts.md` (especially C1 and C6) and `docs/TASKS.md` for context.

To measure how accurate our segmentation is (Dice), every layer needs a reference to compare against. Our evaluation dataset is **ReMIND** (TCIA, CC BY 4.0), stored locally in `data/archive/remind/`. It already ships reference outlines (DICOM SEG) for:
- tumor (manual)
- cerebrum (automatic, from Brainlab), so skull/scalp = head minus cerebrum
- ventricles (automatic, from Brainlab)

**It has nothing for gray matter or white matter. It also has nothing for the CSF outside the ventricles:** the fluid in the brain's folds (sulci) and in the thin layer between the brain and the skull. We call that **outer CSF**.
**Your job is to fill those gaps, and only those gaps, with SynthSeg.**

## Scope (strict)

- Produce **gray-matter, white-matter and outer-CSF reference masks only**.
- Do **not** produce or replace tumor, cerebrum or **ventricle** references. ReMIND's own SEGs stay the reference for those, even though SynthSeg also outputs ventricles.
  The app's `csf` layer = ReMIND ventricles + SynthSeg outer CSF.
- SynthSeg runs **offline only**. Its outputs are used for validation **and as training labels for the AI segmenter** (`docs/contracts.md` C6).
  SynthSeg itself never becomes part of the app's runtime.
- Call the output a **"SynthSeg reference (silver standard)"**, never "ground truth", in files, READMEs and reports.

## Ground rules

- **Read-only on the original data.** Never modify, move or delete anything in `data/archive/remind/<PatientID>/`. Write only under `data/archive/remind/derivatives/synthseg/`.
- `data/archive/remind/` is gitignored, so outputs stay out of git. Scripts go in `scripts/synthseg/`. **Don't commit or push anything without asking your human.**
- **Ask your human before installing anything large**: Docker, WSL distros, FreeSurfer, TensorFlow. Explain what you'll install and how big it is.
- Downloaded archives and DICOMs are untrusted data. Run Python with `python -I`, and keep scripts outside the data folder.
- Don't upload any of this data anywhere.
- If `data/archive/remind/` is missing or incomplete on your machine:
  - Get it with `scripts/analysis/download_remind.py`. It downloads the Preop MR + SEG series of all 114 patients from the public TCIA API, is resumable, and writes `manifest.csv` + `README.md`.
  - If the script isn't in the repo yet, ask your human for it.
  - Check `manifest.csv`: every row should have `status == ok`.

## Step 1: Pick one reference series per patient

The data is laid out as `<PatientID>/Preop/<SeriesNumber>_<Modality>_<Description>/*.dcm`.

1. Use `3D_AX_T1_postcontrast` as the reference series, because ReMIND's cerebrum and ventricle SEGs mostly reference it.
2. If a patient has several of these, pick the one whose `SeriesInstanceUID` is referenced by the patient's cerebrum SEG (`ReferencedSeriesSequence`). Otherwise take the one with the most slices.
3. If a patient has none, fall back in this order: `3D_AX_T1_precontrast` → `3D_SAG_T1_MP2RAGE` → `3D_AX_T2_SPACE` → `3D_SAG_T2_FLAIR`. Mark the case `fallback_series=true`.
4. Skip 2D-only patients and log them.
5. Log every choice in the manifest (Step 6).

## Step 2: Set up the tools (ask before installing)

- **DICOM → NIfTI:** `dcm2niix`. It's a small standalone binary; pin its version.
- **SynthSeg, preferred route:** FreeSurfer ≥ 7.4, through `mri_synthseg` (SynthSeg 2.0).
  - On Windows, run it with the Docker image `freesurfer/freesurfer:7.4.1`, or inside WSL2.
  - FreeSurfer needs a free `license.txt` from https://surfer.nmr.mgh.harvard.edu/registration.html. Your human registers for it.
  - Mount the data read-only, and mount the output dir read-write.
- **SynthSeg, fallback route:** the standalone repo https://github.com/BBillot/SynthSeg, in its own virtual environment, with the Python/TensorFlow versions its README pins.
- **DICOM SEG → mask:** `highdicom` + `pydicom`, or dcmqi's `segimage2itkimage`. Use **SimpleITK** or **nibabel** for resampling.
- Put the Python dependencies in a dedicated virtual environment, and record exact versions in `scripts/synthseg/requirements.txt`.

## Step 3: Run SynthSeg

Run this per patient (adjust paths and threads):

```
dcm2niix -z y -f image -o <out>/<PatientID>/ <reference series dir>
mri_synthseg --i <out>/<PatientID>/image.nii.gz --o <out>/<PatientID>/synthseg_1mm.nii.gz \
             --robust --vol <out>/<PatientID>/volumes.csv --qc <out>/<PatientID>/qc.csv --threads 4
```

- `--robust` is the recommended mode for clinical scans of varying contrast and resolution. **Don't** use `--parc`, since cortical parcels aren't needed.
- If `dcm2niix` produces more than one NIfTI, or flags slices as non-uniform, stop and log it. Never pick one silently.
- **Make it resumable.** Write a `.done` marker per patient, and skip patients that already have one.
- **Run it in the background.** Expect minutes per scan on CPU.
- **Do a pilot first:** run 3 patients end to end (all steps, including the QC images), and **show your human the QC PNGs before batching all of them.**

## Step 4: Bring the labels back onto the native grid

SynthSeg writes its output at 1 mm isotropic resolution, in the same world space as the input.

1. Resample `synthseg_1mm.nii.gz` onto the grid of `image.nii.gz` with **nearest-neighbor** interpolation, using world coordinates (the affines). Save it as `synthseg_native.nii.gz`.
2. Never use linear interpolation on labels.
3. Check alignment visually in the QC PNGs: the label edges must sit on the anatomy.

## Step 5: Apply the label mapping v2 (GM/WM + outer CSF) and build the ignore mask

Write the mapping to `derivatives/synthseg/label_mapping_v2.json` and apply it exactly. (v1 was GM/WM only. If you already wrote v1, keep it and add v2.) Check these label IDs against the SynthSeg/FreeSurfer label table for the version you run. If any differ, stop and tell your human.

| Output | SynthSeg labels |
|---|---|
| `white_matter` | 2, 41 (cerebral WM), 7, 46 (cerebellar WM) |
| `gray_matter` | 3, 42 (cerebral cortex), 8, 47 (cerebellar cortex), 10, 49 (thalamus), 11, 50 (caudate), 12, 51 (putamen), 13, 52 (pallidum), 17, 53 (hippocampus), 18, 54 (amygdala), 26, 58 (accumbens) |
| `csf_outer` | 24 (CSF outside the ventricles: sulci and around the brain) |
| `ignore` (excluded from Dice) | 16 (brainstem), 28, 60 (ventral DC): mixed GM/WM, not fairly scoreable |
| unused (ReMIND covers these) | 4, 5, 14, 15, 43, 44 (ventricles) |

**The ignore mask must also contain the tumor.** SynthSeg wasn't trained on tumors and mislabels tumor and edema as normal tissue. So:

1. Load the patient's ReMIND **tumor** SEG. It may reference a different series than your reference, e.g. ReMIND-001's tumor SEG references `3D_SAG_T2_SPACE`.
2. Convert it to a mask on its own referenced series.
3. Resample it onto the `image.nii.gz` grid with nearest-neighbor interpolation, through world coordinates.
4. **Dilate it by 3 mm.**
5. OR it into `ignore`.
6. Check that the tumor SEG's `FrameOfReferenceUID` matches the reference series. If it doesn't, or there's no tumor SEG, set `tumor_excluded=false` in the manifest. Don't guess.
7. Also add any `previous_resection_cavity` SEG to `ignore` the same way.

Do **not** remove the ignore region from `gray_matter` / `white_matter` / `csf_outer`. Keep those masks as SynthSeg produced them, and keep `ignore` as a separate mask. The Dice code and the AI training code apply it later.

**Save per patient, all on the `image.nii.gz` grid:**
- `gray_matter.nii.gz`, `white_matter.nii.gz` and `csf_outer.nii.gz`: uint8, 0/1
- `ignore.nii.gz`: uint8, 0/1

## Step 6: QC and sanity checks

**QC images.** For each patient, write `derivatives/synthseg/qc_png/<PatientID>.png`:
- 3 axial slices (low, mid, high), plus the slice with the largest tumor area
- columns: image | GM (one color) + WM (another) + outer CSF (a third) overlay | ignore overlay

**Manifest.** Write `derivatives/synthseg/manifest.csv` with these columns:

```
PatientID, reference_series_dir, reference_SeriesInstanceUID, fallback_series, n_nifti, synthseg_qc_min,
gm_cm3, wm_cm3, csf_outer_cm3, tumor_excluded, cavity_excluded, frac_cerebral_gmwm_inside_remind_cerebrum,
frac_gmwm_overlapping_remind_ventricles, frac_csf_outer_overlapping_remind_ventricles, status, notes
```

- Volumes come from the native-grid masks (voxel count × voxel volume), rounded to 0.1 cm³.
- `frac_cerebral_gmwm_inside_remind_cerebrum`: use only labels 2, 3, 41 and 42, since cerebrum excludes cerebellum and brainstem. Expect it to be high.
- `frac_gmwm_overlapping_remind_ventricles`: expect it to be near 0.
- `frac_csf_outer_overlapping_remind_ventricles`: expect it to be near 0 too (outer CSF and ventricles shouldn't overlap).
  Outer CSF is **not** expected to sit inside the ReMIND cerebrum, since much of it lies between the brain and the skull.

**Flags.** Set `status=review` (don't drop the case) when any of these hold:
- the SynthSeg QC score is low (use the threshold its docs suggest; ~0.65 for SynthSeg 2.0, but confirm)
- the inside-cerebrum fraction is < 0.9
- either ventricle overlap is > 0.05
- the volumes (GM, WM or outer CSF) are outliers vs. the cohort median (±3 MAD)

**Determinism.** Rerun 2 patients from scratch and confirm the outputs are byte-identical, or at least label-identical. Record the result.

**Provenance.** Write `derivatives/synthseg/<PatientID>/provenance.json` containing:
- tool versions (dcm2niix, FreeSurfer/SynthSeg, Python packages)
- the exact commands
- the reference `SeriesInstanceUID`
- the tumor/cavity SEG UIDs used
- sha256 of every output file
- the run date

## Step 7: Final layout

```
data/archive/remind/derivatives/synthseg/
  README.md                  # what this is, silver-standard wording, mapping, how to use ignore, citations
  label_mapping_v2.json      # (+ label_mapping_v1.json if you already made it)
  manifest.csv
  qc_png/<PatientID>.png
  <PatientID>/
    image.nii.gz             # dcm2niix of the reference series (the native grid for everything here)
    synthseg_1mm.nii.gz      # raw SynthSeg output, untouched
    synthseg_native.nii.gz   # NN-resampled to image grid
    gray_matter.nii.gz       # uint8 0/1, mapping v2
    white_matter.nii.gz      # uint8 0/1, mapping v2
    csf_outer.nii.gz         # uint8 0/1, mapping v2 (SynthSeg label 24)
    ignore.nii.gz            # uint8 0/1: brainstem + ventral DC + dilated tumor (+ cavity)
    volumes.csv  qc.csv      # SynthSeg's own outputs
    provenance.json
    .done
```

**Add a short "Derivatives: SynthSeg GM/WM + outer-CSF reference" section to `data/archive/remind/README.md`.** It should point to the folder above and state:
- GM, WM and outer CSF only, silver standard
- tumor excluded through `ignore`
- the remaining layers (tumor, cerebrum, ventricles) use ReMIND's own SEGs

**How downstream code will use this.** This is out of scope for you, but make it easy:
- **Gray/white Dice:** our mask vs. `gray_matter` / `white_matter`, computed only on pixels inside the ReMIND cerebrum mask and outside `ignore`.
- **CSF Dice:** our `csf` mask vs. the reference **ReMIND ventricles ∪ `csf_outer`**, outside `ignore`. Ventricles and outer CSF are also reported separately.
- **Skull reference:** head − ReMIND cerebrum − `csf_outer`, so the fluid around the brain no longer counts as skull.
- **Tumor Dice:** vs. ReMIND's tumor SEG.
- **AI training:** the same masks become training labels, with ReMIND taking priority wherever both have a label (tumor > ventricles > SynthSeg).

## Citations to put in the README

- **ReMIND (data):** Juvekar P. et al., *ReMIND: The Brain Resection Multimodal Imaging Database*, TCIA 2023, https://doi.org/10.7937/3RAG-D070
- **ReMIND (paper):** Juvekar P. et al., *Scientific Data* 2024, https://doi.org/10.1038/s41597-024-03295-z
- **SynthSeg:** Billot B. et al., "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining", *Medical Image Analysis* 2023. Robust version: Billot B. et al., *PNAS* 2023. Verify both citations from the SynthSeg README.
- **FreeSurfer:** https://surfer.nmr.mgh.harvard.edu (license terms apply to the software, not to our derived masks).

## Done when

- [ ] The pilot QC PNGs for 3 patients have been approved by your human.
- [ ] Every eligible patient has `status` ok or review. Every skipped patient is logged with a reason.
- [ ] The manifest, mapping JSON, provenance files and README section are written.
- [ ] The determinism check is recorded.
- [ ] You've reported the counts (ok / review / skipped / fallback series / tumor not excluded), the cohort GM, WM and outer-CSF volume medians, anything surprising, and which scripts in `scripts/synthseg/` should be committed. Ask before committing.
