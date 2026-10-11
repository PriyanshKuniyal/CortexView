# SynthSeg reference masks for ReMIND (silver standard)

These masks are a **SynthSeg reference (silver standard)**, not ground truth. They're automated segmentations, used only to score CortexView's classical gray/white-matter segmentation (Dice). Report them as "agreement with a SynthSeg reference", never as accuracy against ground truth.

## Data

- **ReMIND**, Preop studies only, from NCI Imaging Data Commons (`collection_id = 'remind'`).
- 53 patients whose `3D_AX_T1_postcontrast` series has both a tumor SEG and a cerebrum SEG drawn on that same series.
- Optional masks: ventricles (31 patients) and previous resection cavity (12 patients), on the same series.
- Every mask shares the T1's frame of reference, so no registration is involved.

## Per-patient files (native T1 grid, uint8 0/1)

| File | Content |
|---|---|
| `gray_matter.nii.gz` | SynthSeg cortex and deep gray matter: 3, 42, 8, 47, 10, 49, 11, 50, 12, 51, 13, 52, 17, 53, 18, 54, 26, 58 |
| `white_matter.nii.gz` | SynthSeg cerebral and cerebellar white matter: 2, 41, 7, 46 |
| `ignore.nii.gz` | SynthSeg brainstem (16) and ventral DC (28, 60), plus the ReMIND tumor dilated by 3 mm, plus the resection cavity dilated by 3 mm |
| `synthseg_native.nii.gz` | Full SynthSeg label map, resampled (nearest neighbor) from SynthSeg's 1 mm output onto the T1 grid |
| `tumor/cerebrum/ventricles/cavity.nii.gz` | ReMIND manual masks on the T1 grid |

Label numbers follow the FreeSurfer LUT. The full mapping is in `label_mapping_v1.json`. SynthSeg's 33 output labels were checked against FreeSurfer 7.4.1's `synthseg_segmentation_labels_2.0.npy`, and they match the mapping exactly.

### How to use `ignore`

`ignore` is a separate mask. It is **not** subtracted from `gray_matter` or `white_matter`. When scoring, exclude `ignore` voxels from both the prediction and the reference before computing Dice:

```
valid = ~ignore
dice(pred_gm & valid, ref_gm & valid)
```

The tumor and cavity are excluded because SynthSeg's labels there aren't meaningful. Brainstem and ventral DC are excluded because they're mixed GM/WM tissue.

## Known limitations

- **ReMIND's cerebrum mask is tighter than SynthSeg's cerebral GM+WM.** It excludes sulcal CSF and the tumor. On the pilot, only 77–85% of SynthSeg's cerebral GM/WM falls inside it, but 94–99% falls within 2 mm of it. Most of the difference is cortex at the sulcal boundary. Treat the "fraction inside" metric as a sanity check for geometry, not a quality score.
- **Skull/scalp evaluation:** "head mask minus ReMIND cerebrum" is wrong on low slices, because the cerebrum mask excludes the cerebellum and brainstem. Evaluate skull/scalp only above the cerebellum.
- **ReMIND SEG bit-packing.** 27 of ReMIND's 149 selected SEG files (24 patients) were written mis-packed by QIICR dcmqi (`SoftwareVersions ef9e227`). This happens when Rows×Columns mod 8 is 3, 5, 6 or 7. Standard decoders (pydicom, highdicom) return those frames shifted sideways and wrapped. `remind_synthseg.py prepare` repairs them, and each repair is listed in `prepare_notes.json`. See `decode_binary_seg_frames` in the script for the exact layout.
- SynthSeg QC threshold: 0.65 was used as specified. It could not be confirmed in the SynthSeg README.

## Tools

- FreeSurfer 7.4.1 `mri_synthseg --robust`, run on native Windows on CPU.
  - Only the script and model files were taken from the official tarball; see `scripts/synthseg/mri_synthseg_native.py`.
  - Python 3.11, TensorFlow 2.15.1. Exact versions are in `scripts/synthseg/requirements.txt`.
- dcm2niix, highdicom, pydicom, SimpleITK, nibabel.

## Citations

- **ReMIND:** Juvekar P, et al. *ReMIND: The Brain Resection Multimodal Imaging Database.* Scientific Data 11, 494 (2024). doi:10.1038/s41597-024-03295-z. TCIA collection doi:10.7937/3RAG-D070. License: CC BY 4.0.
- **SynthSeg:** Billot B, Greve DN, Puonti O, Thielscher A, Van Leemput K, Fischl B, Dalca AV, Iglesias JE. *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining.* Medical Image Analysis 86, 102789 (2023). doi:10.1016/j.media.2023.102789
- **SynthSeg robust, QC:** Billot B, Magdamo C, Cheng Y, Arnold SE, Das S, Iglesias JE. *Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets.* PNAS 120(9), e2216399120 (2023). doi:10.1073/pnas.2216399120

The SynthSeg citations were checked against the SynthSeg GitHub README and `bibtex.bib`. The ReMIND citation is as given in the task brief and was not independently re-checked.
