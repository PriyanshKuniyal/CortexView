# Snapshot of the classical loader

The AI segmenter loads DICOM exactly the way the classical one does, so both get the same `Volume` for the same upload.
These files were **copied unchanged** from the classical work (working tree of `Plie`, branch `shared-contracts`, not yet committed there) on 2026-10-10:

| File | sha256 (first 16) |
|---|---|
| `backend/cortexview/__init__.py` | `6A3B03150805E136` |
| `backend/cortexview/dicom_io.py` | `743C90D0B5CB3738` |
| `backend/cortexview/volume.py` | `E6EAA274110D2170` |
| `backend/cortexview/layers.py` | `62702DECC8797645` |
| `backend/requirements.txt` | `335ACEECBA79A1D2` |

**Don't edit them on this branch.** The classical side owns them.
When the classical branch is committed, merge it in and take its versions; this table only shows what the AI code was built against.
