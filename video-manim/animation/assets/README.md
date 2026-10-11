# Assets

This folder is empty on purpose. The video loads no image files. Every picture is computed at render time:

- The **synthetic slice, label map, layers and shaded mock-up** come from `../phantom.py`. It is a NumPy port of the web prototype's phantom (`frontend/src/cortexview/demo/phantom.ts`), and `../tests/test_phantom_port.py` checks it pixel for pixel against the original.
- **Tissue colors** are read from `shared/layers.json`, the project's frozen palette.

No patient data and no ReMIND images are used. To add an asset (e.g. a logo), put it here and load it with `ImageMobject(str(Path(__file__).parent / "assets" / "file.png"))`.
