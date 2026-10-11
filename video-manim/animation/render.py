"""Render helper for the CortexView explainer: one command per job, outputs in video-manim/output/.

    python video-manim/animation/render.py preview            # 1280x720 @ 30 fps -> output/cortexview_preview.mp4
    python video-manim/animation/render.py final              # 1920x1080 @ 60 fps -> output/cortexview_final.mp4
    python video-manim/animation/render.py scene S3Measure    # one scene, 480p15, for debugging
    python video-manim/animation/render.py mux voiceover.mp3  # final video + narration -> output/cortexview_final_voiceover.mp4

Manim's cache (partial movie files, several hundred MB) goes to $CORTEXVIEW_MEDIA, default
~/cortexview_video/media: outside the repo, which is synced by OneDrive. Run with the Python that has
Manim installed (see README.md). `mux` needs ffmpeg on PATH; it never calls any external API.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "cortexview_video.py"
OUT = HERE.parent / "output"
MEDIA = Path(os.environ.get("CORTEXVIEW_MEDIA", Path.home() / "cortexview_video" / "media"))

QUALITY = {  # job: (manim flags, folder Manim writes to)
    "preview": (["-qm"], "720p30"),
    "final": (["-qh", "--fps", "60"], "1080p60"),
    "scene": (["-ql"], "480p15"),
}


def render(job: str, scene: str = "CortexViewFull") -> Path:
    flags, folder = QUALITY[job]
    cmd = [sys.executable, "-m", "manim", *flags, "--media_dir", str(MEDIA), str(SCRIPT), scene]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return MEDIA / "videos" / SCRIPT.stem / folder / f"{scene}.mp4"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="job", required=True)
    sub.add_parser("preview")
    sub.add_parser("final")
    p_scene = sub.add_parser("scene")
    p_scene.add_argument("name", help="S1Hook, S2Layers, S3Measure, S4Design, S5Status, S6Closing or CortexViewFull")
    p_mux = sub.add_parser("mux")
    p_mux.add_argument("audio", type=Path, help="narration file from ElevenLabs (mp3/wav)")
    p_mux.add_argument("--video", type=Path, default=OUT / "cortexview_final.mp4")
    p_mux.add_argument("--offset", type=float, default=0.0, help="delay the narration by this many seconds")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if a.job in ("preview", "final"):
        src = render(a.job)
        dst = OUT / f"cortexview_{a.job}.mp4"
        shutil.copy2(src, dst)
        print("wrote", dst)
    elif a.job == "scene":
        print("wrote", render("scene", a.name))
    else:
        dst = OUT / "cortexview_final_voiceover.mp4"
        # keep the video stream as is; AAC audio; stop at the shorter of the two; optional narration delay
        cmd = ["ffmpeg", "-y", "-i", str(a.video), "-itsoffset", str(a.offset), "-i", str(a.audio),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dst)]
        print(">", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        print("wrote", dst)


if __name__ == "__main__":
    main()
