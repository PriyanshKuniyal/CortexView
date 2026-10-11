"""Plain-language text for the pipeline's warning codes (C3 `warnings`). Never includes header values or file names."""

WARNINGS = {
    "seg_resampled": "A doctor's outline was drawn on a different scan of the same session and was mapped onto this one.",
    "seg_frame_mismatch": "A doctor's outline was drawn on a scan in a different coordinate system, so it wasn't used.",
    "seg_segment_ignored": "A doctor's outline had a structure CortexView doesn't show, so that part was ignored.",
    "seg_decode_suspect": "A doctor's outline couldn't be read reliably, so it wasn't used.",
    "low_slice_count": "Only a few slices were uploaded, so this slice was analyzed on its own and volumes aren't available.",
    "nonuniform_slices": "The slices aren't evenly spaced, so 3D measurements may be less accurate.",
    "sequence_unknown": "The scan type (T1, T2 or FLAIR) couldn't be recognised, so a default was assumed.",
    "noisy_background": "The image is noisy, so the automatic layers are less reliable.",
    "gm_wm_low_contrast": "Gray and white matter look very similar on this scan, so their boundary is approximate.",
    "brain_label_extended": "The doctor's brain outline didn't cover everything, so some of the brain edge was found automatically.",
    "csf_includes_doctor_ventricles": "The fluid layer includes the doctor's ventricle outline.",
    "lesion_from_other_series": "The suspected lesion was found on a FLAIR scan of the same session and mapped onto this one.",
    "lesion_auto_needs_flair": "Automatic lesion detection needs a FLAIR scan, and none was uploaded.",
    "lesion_no_midline": "The brain's midline couldn't be found, so lesion detection used a simpler comparison.",
}


def warning_message(code: str) -> str:
    return WARNINGS.get(code, "Note: " + code.replace("_", " ") + ".")
