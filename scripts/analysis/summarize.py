"""Summarize scan JSON. Usage: python -I summarize.py <scan.json>"""
import json, sys, statistics as st
from collections import Counter, defaultdict
d = json.load(open(sys.argv[1])); R = d["rows"]
by = defaultdict(list)
for r in R: by[r["case"]].append(r)
print("cases", len(by), "patients", len({r["patient"] for r in R}), "slices", len(R))
n = [len(v) for v in by.values()]; print("slices/case min/med/max", min(n), st.median(n), max(n))
gaps = [c for c, v in by.items() if sorted(x["slice"] for x in v) != list(range(1, len(v) + 1))]
print("cases with non-contiguous slice numbering", len(gaps))
print("mask value sets", Counter(tuple(r["mask_values"]) for r in R))
pos = [r for r in R if r["mask_px"]]
print("positive slices", len(pos), f"{len(pos)/len(R):.1%}", "cases with any positive", len({r['case'] for r in pos}))
print("cases w/o any positive", [c for c, v in by.items() if not any(x["mask_px"] for x in v)])
mp = sorted(r["mask_px"] for r in pos); q = lambda a, p: a[int(p * (len(a) - 1))]
print("lesion px/slice p0 p10 p50 p90 p100", [q(mp, p) for p in (0, .1, .5, .9, 1)])
cc = Counter(min(len(r["mask_components"]), 5) for r in pos); print("components per positive slice (5=5+)", sorted(cc.items()))
tiny = sum(1 for r in pos if r["mask_px"] < 50); print("positive slices with <50 px", tiny)
fe = sorted(r["mask_feret_px"] for r in pos if r.get("mask_feret_px")); print("max feret px p50 p90 max", q(fe, .5), q(fe, .9), fe[-1])
contrast = [r["lesion_flair_mean"] / r["nonlesion_flair_mean"] for r in pos if r["nonlesion_flair_mean"] and r["mask_px"] >= 100]
contrast.sort(); print("lesion/non-lesion FLAIR mean ratio p10 p50 p90", [round(q(contrast, p), 2) for p in (.1, .5, .9)], "frac ratio<1.1", round(sum(c < 1.1 for c in contrast) / len(contrast), 3))
# missing sequences per case
pre_missing = [c for c, v in by.items() if all(x["pre_eq_flair"] for x in v)]
post_missing = [c for c, v in by.items() if all(x["post_eq_flair"] for x in v)]
partial = [c for c, v in by.items() if 0 < sum(x["pre_eq_flair"] or x["post_eq_flair"] for x in v) < len(v)]
print("pre-contrast missing (ch0==ch1 all slices)", len(pre_missing), "post missing", len(post_missing), "partial-equal cases", len(partial))
for k, nm in enumerate(["pre", "flair", "post"]):
    mx = Counter(r["ch"][k]["max"] for r in R); mxc = Counter(max(x["ch"][k]["max"] for x in v) for v in by.values())
    print(nm, "per-case max dist", sorted(mxc.items())[-5:], "| slice p99 med", st.median(r["ch"][k]["p99"] for r in R),
          "| slice mean med", round(st.median(r["ch"][k]["mean"] for r in R), 1), "| nonzero frac med", round(st.median(r["ch"][k]["nonzero_frac"] for r in R), 3))
blank = [r for r in R if r["ch"][1]["max"] < 20]; print("near-blank FLAIR slices", len(blank))
print("border nonzero>0 slices", sum(r["border_nonzero"] > 0 for r in R), "median border px", st.median(r["border_nonzero"] for r in R))
inst = Counter(r["case"].split("_")[1] for r in R if r["slice"] == 1); print("institutions (cases)", inst)
print("unique dates", len({r["date"] for r in R}), "min/max", min(r["date"] for r in R), max(r["date"] for r in R))
# laterality via centroid (image x): 
lx = [r["mask_centroid_xy"][0] for r in pos if r["mask_px"] >= 100]; print("lesion centroid x <128 frac", round(sum(x < 128 for x in lx) / len(lx), 2))
