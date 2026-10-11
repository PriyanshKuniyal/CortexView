# CortexView classical segmentation

The segmenter turns a brain MRI into the five C1 layers: `skull`, `gray_matter`, `white_matter`, `csf` and `lesion`.
- **No AI is involved.** Every step is plain math: thresholds, distance maps, connected blobs and robust statistics.
- **It's deterministic.** The same input always gives byte-identical masks; the tests check this.
- **It works in 3D on the whole series,** then cuts out the slice that is shown.
- **Sizes are in mm,** so thick-slice scans (e.g. 5 mm gaps) are handled correctly.

Code: `backend/cortexview/`. Tests: `tests/`. Evaluation scripts: `scripts/seg/`.

## How it works, in plain words

1. **Find the head.**
   - Air is dark. An automatic threshold (Otsu) splits head from air.
   - Then: close small gaps, keep the biggest blob, and fill holes.
   - Some scans (MP2RAGE "UNI" images) fill the air with bright speckle noise. These are detected and flagged (`noisy_background`), because brightness can't find the head on them.
2. **Normalise brightness.**
   - MRI brightness has no unit. Each scan is scaled to its own 1st–99th percentile.
   - The coil's slow brightness drift across the head (bias field) is divided out.
3. **Find the brain boundary** (the intracranial space: brain + CSF + any tumour). Everything else inside the head becomes the `skull` layer (bone, scalp, face).
   1. **Measure typical brain brightness** where brain certainly is: the upper half of the head, more than 25 mm under the skin.
      - "Tissue" is anything brighter than 0.7× that value. This keeps gray and white matter and drops fluid, bone and air.
      - Using this anchor matters. A histogram split of the whole head failed on scans with a lot of bright scalp fat or neck.
   2. **Conservative brain.**
      - Shrink the tissue mask by 6 mm and keep only voxels more than 15 mm deep. This cuts the thin bridges to scalp, eyes and neck.
      - Keep the blob that overlaps the "certainly brain" region, then grow it back.
   3. **Extend to the skull cavity.**
      - The skull is too thin and too variable in brightness to trace directly. On T1 it's a 1–3 mm dark line next to bright marrow, sometimes with no dark line at all.
      - What's stable is geometry: over the top of the head, the inner skull lies at a fairly constant depth under the skin.
      - Upper head: anything deeper than that shell and not bone-dark is added. This recovers the top of the brain, outer CSF, and dark tumours touching the skull.
      - Skull base: only CSF/tumour-dark blobs inside the brain's own outline are added. Sinuses, muscles and orbits stay out.
      - A thin outer-CSF rim is added last.
4. **Lesion.**
   - A doctor label (DICOM-SEG) is always used when present: `source: doctor`, `confidence: high`.
   - Otherwise lesions are found **on FLAIR only**. Swelling and most gliomas are bright on FLAIR, and fluid is dark.
     1. Fit the brain's left/right mirror plane.
     2. **Seeds**: voxels more than 3 robust SDs brighter than typical brain, and more than 2 SDs brighter than their mirror point. Most tumours are one-sided; normally bright anatomy is bright on both sides.
     3. **Growth**: the lesion grows from the seeds through voxels above 2 SDs that are still asymmetric (hysteresis thresholding).
     4. Size and skull-rim filters remove the rest.
   - When the shown series is T1 and a FLAIR of the same scan exists, the FLAIR lesion is mapped onto the T1 (`lesion_from_other_series`).
   - Not done on T1 or T2 (`lesion_auto_needs_flair`):
     - On T1, tumours can be dark, bright or both.
     - On T2, fluid is the brightest thing and swelling overlaps it. Brightness outliers are then mostly CSF edges; tested on ReMIND and the phantom.
5. **CSF / gray matter / white matter.**
   - **CSF** is everything past the halfway point between typical tissue and typical fluid brightness. Fluid is darkest on T1/FLAIR and brightest on T2.
   - **Gray vs white** combines two cues:
     - *Local* contrast: cortex is a little darker than the white matter right next to it on T1, brighter on T2/FLAIR.
     - *Geometry*: gray matter is the ribbon along the brain surface and fluid-filled folds.
   - The scan's measured gray/white contrast decides how much each cue counts. On these clinical scans it is weak (d′ ≈ 0.5–1.2), so geometry leads and confidence is reported as `low`.
6. **One label per pixel.** Priority order: lesion > csf > white_matter > gray_matter > skull.

## Interfaces

| Function | What it does |
|---|---|
| `segment.analyze(img, spacing_mm, sequence, doctor=None, tissue_slices=None, lesion_hint=None)` | Core function. Takes a 2D slice or 3D volume and returns a `Segmentation`: masks, confidence, source, warnings, head, intracranial, stats, `slice_index`. |
| `segment.segment_volume(volume, meta)` | C6 interface, 3D: `{layer_id: bool mask}`. |
| `segment.segment(slice, meta)` | C6 interface, one slice. If `meta["volume"]` is given, it segments in 3D and cuts out `meta["slice_index"]`; otherwise it segments the single slice in 2D. |
| `pipeline.run(files, slice_index=None, series_uid=None)` | The C3 flow, from `.dcm` paths, file objects and/or zip archives to the shown slice. Zips are unpacked in memory, with a 1 GB guard. Raises `AnalyzeError` with the C3 codes (`not_dicom`, `not_mr`, `no_axial_series`, `bad_slice_index`, `too_large`). This is what `POST /api/analyze` should call. |
| `dicom_io.scan` / `choose_series` / `load_info` / `load_seg` / `doctor_labels_on` | Header-only grouping, the C3 series choice, pixel loading, SEG reading, and mapping labels onto the shown grid via patient coordinates. |
| `export.write_slice_folder(...)` | Writes a C2 folder: `meta.json`, `mri.png`, `label.png`, `guide.png`, `masks/`, `reference/`. Padded to square, never resized. |

### `tissue_slices` (speed)
Head, brain boundary and lesion are always computed in 3D over the whole volume. The CSF/GM/WM split is the slow part, so:
- `tissue_slices="auto"` limits it to the C3 auto slice. The API uses this.
- That slice is **identical** to a full run; a test checks this.
- Use `None` (every slice) for offline export or volumes.

### Speed
| Step | Time |
|---|---|
| Whole C3 pipeline, one ReMIND patient (300–700 files, 1 mm 3D T1) | about 6–9 s |
| of which: header scan | under 0.5 s |
| of which: segmentation alone, API mode | about 3–5 s |

The C3 target of under 1 s isn't realistic for 3D classical segmentation in Python. Run analysis inside the background job the frontend already polls (see open decision C6).

## Warnings

| Code | Meaning |
|---|---|
| `seg_frame_mismatch` | A doctor label was drawn on a scan in another coordinate frame, so it isn't used. Patients can move between series; see the data findings. |
| `seg_resampled` | A doctor label was mapped onto the shown grid (nearest neighbour, patient coordinates). |
| `seg_decode_suspect` | A doctor label decodes as sheared/wrapped frames, so it isn't used (see the data findings). |
| `seg_segment_ignored` | A SEG segment that isn't tumour, ventricles or brain, e.g. a resection cavity. |
| `brain_label_extended` | The doctor "cerebrum" label was combined with the automatic boundary, because cerebrum labels leave out the cerebellum, brainstem and outer CSF. |
| `csf_includes_doctor_ventricles` | Doctor ventricles were added to CSF. |
| `lesion_auto_needs_flair` | No doctor lesion, and the shown series isn't FLAIR (and there's no FLAIR in the same frame). The lesion layer is empty. |
| `lesion_from_other_series` | The lesion was found on a FLAIR of the same scan and mapped onto the shown series. |
| `lesion_no_midline` | The left/right mirror couldn't be fitted, so stricter brightness-only seeds were used. |
| `gm_wm_low_contrast` | This scan barely separates gray and white matter, so GM/WM confidence is `low`. |
| `noisy_background` | Speckle-noise background (e.g. MP2RAGE). The automatic layers are unreliable and all are marked `low`. |
| `low_slice_count` | Fewer than 20 slices, so the slice was segmented on its own (2D). |
| `tissue_partial` | CSF/GM/WM were computed only for the requested slices (API mode). |
| `nonuniform_slices`, `orientation_ambiguous`, `sequence_unknown`, `no_head_found` | Input problems, as named. |

## Confidence
Each layer starts at a ceiling set by the sequence; the ceiling is then lowered by what the scan actually shows:
- **GM/WM**: from the measured gray/white contrast d′. Under 1.0 is `low`, 1.0–2.0 `medium`, above 2.0 `high`.
- **CSF**: from how far fluid sits from tissue, in robust SDs. Under 3 is `low`, 3–5 `medium`, above 5 `high`.
- **Skull**: `high` when a doctor brain label defines the boundary; `medium` when it had to be extended with the automatic one.
- **Lesion**: `high` for doctor labels, `medium` for FLAIR auto lesions.
- **Empty**: any layer with no pixels on the shown slice is `empty`.

## Validation

All references are taken only from SEG files in the **same coordinate frame** as the evaluated series, and damaged files are excluded.
- Brain-boundary scores are against BrainLab's *cerebrum* label (+ tumour + ventricles).
- That label **excludes the cerebellum, brainstem and the CSF around the brain**, which our boundary rightly includes. So precision and Dice understate the real agreement.
- *Recall* (how much of the reference brain we capture) is the cleanest number.

### Synthetic phantom (known answer, `tests/`)
| Sequence | Brain | Skull | GM | WM | CSF | Lesion |
|---|---|---|---|---|---|---|
| T1-post | 0.98 | 0.92 | 0.88 | 0.98 | 0.74 | (none on T1) |
| T2 | 0.97 | 0.92 | 0.77 | 0.96 | 0.87 | (none on T2) |
| FLAIR | 0.98 | 0.94 | 0.86 | 0.97 | 0.80 | 0.81 |

All values are Dice.

### ReMIND (45 patients downloaded; fully automatic, no doctor labels given to the segmenter)
**T1: brain boundary, 27 scans.** These are the scans with a usable same-frame cerebrum label. One MP2RAGE scan is flagged `noisy_background` and excluded.

| | Mean | Median | Worst |
|---|---|---|---|
| Brain recall (share of the reference brain we include) | **0.97** | 0.99 | 0.86 |
| Brain Dice | 0.74 | 0.74 | 0.64 |

- **Volume:** our intracranial volume has a median of 1,497 mL (range 1,124–1,954). The reference cerebrum label alone has a median of 926 mL.
- **Ventricles:** 60% of reference ventricle voxels are labelled CSF (19 scans). The reference includes the enhancing choroid plexus, which isn't fluid.
- **Before the fixes**, the worst scan scored Dice 0.25, and five scans leaked into the face/neck (1.8–2.9 L).

**T2, 41 scans.** No same-frame brain labels exist (they're drawn on T1), so this is a robustness check:
- Every scan ran.
- Boundary volume: median 1,633 mL (1,238–2,123). It's higher than T1 because bright outer CSF is correctly included.
- Visual checks are clean, including 5 mm-slice and coronal series.

**FLAIR lesion, the 4 scans with a same-frame tumour label:**

| Case | Dice | Found vs reference |
|---|---|---|
| ReMIND-026 | 0.83 | 49 vs 68 mL |
| ReMIND-040 | 0.71 | 12 vs 11 mL |
| ReMIND-042 | 0.33 | 18 vs 15 mL |
| ReMIND-012 | 0.22 | subtle, 4 vs 5 mL |
| **Mean** | **0.52** | |

### 2D fallback (single slice, used for uploads with fewer than 20 slices)
- Compared with the 3D result for the same slice on ReMIND T1/T2/FLAIR:
  - brain-boundary Dice **0.95–0.99** on most slices
  - weakest: 0.78 on a high slice of a 0.47 mm scan, and 0.68 on a low slice of a 5 mm FLAIR
- On the phantom: 0.99.

### Kaggle LGG (110 patients, FLAIR lesion only; evaluation only, never trained on)
| | |
|---|---|
| Lesion Dice | mean **0.59**, median 0.61 (middle half 0.46–0.75) |
| Patients above 0.5 | 66% |
| Complete misses | 1% |
| Slice detection | sensitivity 0.96, specificity 0.76 |

- Voxel size isn't recorded in this set, so it is estimated from head size.
- For scale: the U-Net published with this dataset reaches about 0.82, but it was trained on this data.
- Most false positives are fat/tissue around the eyes and at the skull base on thick-slice scans.

## Findings about the data

1. **ReMIND series don't reliably share a coordinate frame.** After anonymisation every series has its own `FrameOfReferenceUID`.
   - Some pairs still line up (ReMIND-002: two tumour labels agree at Dice 0.89).
   - Others clearly don't (ReMIND-003's 2D T2 vs its T1: the patient moved).
   - **Conclusion: the strict C3 rule (only use a label in the same frame) is correct.**
2. **Some ReMIND DICOM-SEG files decode as sheared frames.**
   - About 1 in 5 of the files checked, e.g. the cerebrum labels of 004/010/012/037/044 and several tumour/ventricle labels.
   - Read per the standard, each 1-bit frame is shifted sideways and wraps around the frame edge.
   - It happens only when a frame's pixel count isn't a multiple of 8 bits (remainders 3, 5, 6, 7), so the original writer packed frames differently. The exact layout couldn't be recovered with certainty.
   - These files are detected (centre drifts sideways more than 1.5 mm per 1.5 mm of depth, *and* over 20% of frames wrap) and **not used** (`seg_decode_suspect`), rather than painting a garbled label.
3. **BrainLab's "cerebrum" label cuts out the tumour and ventricles and leaves out the cerebellum,** so it's combined with the automatic boundary.
4. **TR/TE/TI are blank in ReMIND headers,** so sequence detection relies on `SeriesDescription`.
5. **Post-contrast T1 in ReMIND has weak gray/white contrast** (d′ 0.2–1.1). This is a property of these navigation scans, not something tuning can fix.
6. **MP2RAGE series (ReMIND-012/023/040/044) have speckle-noise backgrounds.** They're flagged; C3 prefers other series anyway.

## Known limits
- **No automatic lesion on T1 or T2.** The lesion layer is empty unless there is a doctor label or a FLAIR in the same frame.
- **FLAIR auto-lesions** over-call fat and tissue near the orbits/skull base on thick-slice scans, and miss very subtle lesions (e.g. ReMIND-012).
- **GM/WM is geometry-led on low-contrast scans.** Deep gray nuclei (thalamus, basal ganglia) come out as white matter.
- **The skull base is the least accurate region** (sinuses, petrous bone). The display slice is chosen inside the brain, so it's rarely shown.
- **MP2RAGE-style noisy backgrounds** aren't segmented reliably (flagged).

## Running it
```
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
cd tests && ../backend/.venv/Scripts/python.exe -m unittest            # 49 tests, ~2.5 min
backend/.venv/Scripts/python.exe -I scripts/seg/run_on_remind.py --patients 1-45 --sequence T1   # T1 | T2 | FLAIR
backend/.venv/Scripts/python.exe -I scripts/seg/eval_kaggle.py
backend/.venv/Scripts/python.exe -I scripts/seg/export_remind.py --patients 2,7,13,16,25 --out data/out/slices --require-lesion
```
Outputs go to `data/out/` (gitignored). Exported C2 folders use anonymous IDs (`s0001`, …), and the private ID map is written to `~/.cortexview/id_map.csv`, outside the repo.

## Proposed contract updates (for PK; not applied)
- **C3**
  - Add the warning codes above.
  - Note that the analysis takes about 6–9 s, so it belongs in the background job.
  - Optional: show a FLAIR lesion on the T1 display (`lesion_from_other_series`).
- **C6**
  - `segment()` gets the volume via `meta["volume"]`.
  - `analyze(..., tissue_slices="auto")` is the fast path.
- **C2**
  - Exported `meta.json` carries extra `x_layer_source`, `x_layer_confidence` and `x_warnings` fields. They're prefixed so they never clash.
