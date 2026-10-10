"""Download the Preop MR + SEG subset of the TCIA ReMIND collection (CC BY 4.0).

Standard library only. Run with `python -I`.

Usage:
    python -I download_remind.py --out <target_root> [--workers 4] [--tmp <tmpdir>]

Layout: <out>/<PatientID>/<StudyDesc>/<SeriesNumber>_<Modality>_<desc>/*.dcm
Resumable: each finished series folder gets a `.complete` marker and is skipped later.
Downloaded archives are treated as untrusted data: extracted only (never executed),
entries with absolute paths / drive letters / '..' are rejected.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
SERIES_URL = API + "/getSeries?Collection=ReMIND"
IMAGE_URL = API + "/getImage?SeriesInstanceUID={uid}"
UA = {"User-Agent": "remind-downloader/1.0 (python urllib; research use)"}

log_lock = threading.Lock()


def log(msg):
    with log_lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def sanitize(s, maxlen=60):
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", s or "")
    s = re.sub(r"_+", "_", s).strip("_.")
    return (s[:maxlen].rstrip("_.")) or "noname"


def fetch_series():
    req = urllib.request.Request(SERIES_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def select(series):
    sel = [s for s in series
           if s.get("StudyDesc") == "Preop" and s.get("Modality") in ("MR", "SEG")]
    sel.sort(key=lambda s: (s["PatientID"], s.get("StudyDesc", ""),
                            int(s.get("SeriesNumber") or 0), s["SeriesInstanceUID"]))
    # assign relative paths; disambiguate collisions with a UID suffix
    seen = {}
    for s in sel:
        name = "{}_{}_{}".format(s.get("SeriesNumber", "NA"), s["Modality"],
                                 sanitize(s.get("SeriesDescription")))
        rel = "/".join([sanitize(s["PatientID"]), sanitize(s["StudyDesc"]), name])
        if rel in seen:
            rel = rel + "_" + s["SeriesInstanceUID"].split(".")[-1][-8:]
        seen[rel] = True
        s["_rel"] = rel
    return sel


def safe_extract(zpath, dest):
    """Extract only regular files, rejecting unsafe names. Returns list of written files."""
    written = []
    dest_abs = os.path.abspath(dest)
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            parts = name.split("/")
            if (name.startswith("/") or re.match(r"^[A-Za-z]:", name)
                    or ".." in parts or any(":" in p for p in parts)):
                raise ValueError("unsafe zip entry: %r" % info.filename)
            # flatten: series zips are flat anyway; keep only basename
            base = sanitize(parts[-1], maxlen=120)
            target = os.path.abspath(os.path.join(dest_abs, base))
            if os.path.dirname(target) != dest_abs:
                raise ValueError("unsafe zip entry: %r" % info.filename)
            if os.path.exists(target):
                raise ValueError("duplicate zip entry name: %r" % info.filename)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            written.append(target)
    return written


def download(uid, tmpdir):
    fd, zpath = tempfile.mkstemp(suffix=".zip", dir=tmpdir)
    try:
        req = urllib.request.Request(IMAGE_URL.format(uid=uid), headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r, os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except BaseException:
        try:
            os.remove(zpath)
        except OSError:
            pass
        raise
    return zpath


def dir_stats(d):
    n = b = 0
    for e in os.scandir(d):
        if e.is_file() and e.name.lower().endswith(".dcm"):
            n += 1
            b += e.stat().st_size
    return n, b


def process(s, out, tmpdir, retries=3):
    final = os.path.join(out, *s["_rel"].split("/"))
    marker = os.path.join(final, ".complete")
    expected = int(s.get("ImageCount") or 0)
    if os.path.isfile(marker):
        n, b = dir_stats(final)
        return s, n, b, "ok" if n == expected else "mismatch", "resumed"
    last_err = None
    for attempt in range(1, retries + 1):
        zpath = None
        staging = None
        try:
            zpath = download(s["SeriesInstanceUID"], tmpdir)
            staging = tempfile.mkdtemp(prefix="x_", dir=tmpdir)  # new, empty dir
            safe_extract(zpath, staging)
            n, b = dir_stats(staging)
            if n == 0:
                raise RuntimeError("no .dcm files in archive")
            status = "ok" if n == expected else "mismatch"
            if os.path.isdir(final):
                shutil.rmtree(final)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            shutil.move(staging, final)
            staging = None
            with open(marker, "w") as f:
                json.dump({"SeriesInstanceUID": s["SeriesInstanceUID"], "files": n,
                           "expected": expected, "bytes": b, "status": status,
                           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
            if status != "ok":
                log("MISMATCH %s: %d files vs ImageCount %d" % (s["_rel"], n, expected))
            return s, n, b, status, "downloaded"
        except Exception as e:  # noqa: BLE001
            last_err = e
            log("attempt %d/%d failed for %s: %s: %s" % (attempt, retries, s["_rel"],
                                                        type(e).__name__, e))
            if attempt < retries:
                time.sleep(15 * 3 ** (attempt - 1))  # 15 s, 45 s
        finally:
            if zpath and os.path.exists(zpath):
                os.remove(zpath)
            if staging and os.path.isdir(staging):
                shutil.rmtree(staging, ignore_errors=True)
    return s, 0, 0, "failed", repr(last_err)


def write_manifest(out, sel, results):
    """Write a row for every selected series; rows not processed yet are 'pending'."""
    path = os.path.join(out, "manifest.csv")
    tmp = path + ".tmp"
    by_uid = {r[0]["SeriesInstanceUID"]: r for r in results}
    cols = ["PatientID", "StudyDesc", "Modality", "SeriesNumber", "SeriesDescription",
            "SeriesInstanceUID", "ImageCount", "files_on_disk", "bytes_on_disk",
            "relative_path", "status"]
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in sorted(sel, key=lambda x: x["_rel"]):
            _, n, b, status, _ = by_uid.get(s["SeriesInstanceUID"], (s, 0, 0, "pending", ""))
            w.writerow([s["PatientID"], s["StudyDesc"], s["Modality"], s.get("SeriesNumber"),
                        s.get("SeriesDescription"), s["SeriesInstanceUID"],
                        s.get("ImageCount"), n, b, s["_rel"], status])
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None, help="temp dir for zips (default: %TEMP%/remind_dl_tmp)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only first N series (testing)")
    ap.add_argument("--passes", type=int, default=2,
                    help="extra passes re-try series that failed all retries in earlier passes")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    tmpdir = os.path.abspath(a.tmp or os.path.join(tempfile.gettempdir(), "remind_dl_tmp"))
    os.makedirs(tmpdir, exist_ok=True)

    series = fetch_series()
    sel = select(series)
    if a.limit:
        sel = sel[:a.limit]
    tot_bytes = sum(int(s.get("FileSize") or 0) for s in sel)
    by_mod = {}
    for s in sel:
        by_mod[s["Modality"]] = by_mod.get(s["Modality"], 0) + 1
    log("collection series: %d; selected: %d %s; patients: %d; expected %.2f GB, %d images"
        % (len(series), len(sel), by_mod, len({s['PatientID'] for s in sel}),
           tot_bytes / 1e9, sum(int(s.get('ImageCount') or 0) for s in sel)))
    if a.dry_run:
        return

    res = {}  # uid -> result tuple
    t0 = time.time()
    todo = sel
    for p in range(1, max(1, a.passes) + 1):
        if not todo:
            break
        workers = max(1, min(a.workers, 4)) if p == 1 else 2
        if p > 1:
            log("pass %d: retrying %d failed series with %d workers after 60 s"
                % (p, len(todo), workers))
            time.sleep(60)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process, s, out, tmpdir) for s in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                res[r[0]["SeriesInstanceUID"]] = r
                if i % 10 == 0 or r[3] != "ok":
                    gb = sum(x[2] for x in res.values()) / 1e9
                    log("[pass %d %d/%d] %s %s | %.2f GB on disk, %.1f min"
                        % (p, i, len(todo), r[3], r[0]["_rel"], gb, (time.time() - t0) / 60))
                if i % 25 == 0:
                    write_manifest(out, sel, list(res.values()))
        todo = [s for s in sel if res[s["SeriesInstanceUID"]][3] == "failed"]
    results = list(res.values())
    mpath = write_manifest(out, sel, results)
    counts = {}
    for r in results:
        counts[r[3]] = counts.get(r[3], 0) + 1
    log("DONE. statuses: %s; manifest: %s" % (counts, mpath))
    for r in results:
        if r[3] != "ok":
            log("  %s %s (%s) ImageCount=%s files=%d %s" % (r[3], r[0]["_rel"],
                r[0]["SeriesInstanceUID"], r[0].get("ImageCount"), r[1], r[4]))
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
