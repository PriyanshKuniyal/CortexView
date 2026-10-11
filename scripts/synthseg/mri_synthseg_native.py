"""
Run FreeSurfer 7.4.1's `mri_synthseg` on native Windows (no WSL, no FreeSurfer install).

Only two pieces of FreeSurfer are needed, both extracted from the official tarball
(freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz):
    freesurfer/python/scripts/mri_synthseg
    freesurfer/models/synthseg_*

`mri_synthseg` uses surfa only for `sf.system.fatal`, so a stub replaces it here, and numpy's missing
`float128` (absent on Windows) is aliased. The FreeSurfer script itself is not modified.
Needs tensorflow 2.15 (Keras 2) and numpy < 2.

    python mri_synthseg_native.py --fs-home C:/path/to/freesurfer -- --i in.nii.gz --o out.nii.gz --robust ...
"""
import os
import runpy
import sys
import types


def main():
    argv = sys.argv[1:]
    if "--fs-home" not in argv or "--" not in argv:
        sys.exit(__doc__)
    fs_home = argv[argv.index("--fs-home") + 1]
    rest = argv[argv.index("--") + 1:]

    def fatal(msg, retcode=1):
        print(f"Error: {msg}", file=sys.stderr, flush=True)
        sys.exit(retcode)

    surfa = types.ModuleType("surfa")
    surfa.system = types.SimpleNamespace(fatal=fatal)
    sys.modules["surfa"] = surfa

    # Windows numpy has no float128; mri_synthseg only names it in an isinstance() check
    import numpy as np
    if not hasattr(np, "float128"):
        np.float128 = np.longdouble

    os.environ["FREESURFER_HOME"] = fs_home
    script = os.path.join(fs_home, "python", "scripts", "mri_synthseg")
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
