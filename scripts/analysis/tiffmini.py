"""Minimal read-only TIFF reader (stdlib only) for uncompressed, single-strip/multi-strip 8-bit TIFFs."""
import struct

TAGS = {256: "width", 257: "height", 258: "bits", 259: "compression", 262: "photometric",
        273: "strip_offsets", 277: "spp", 278: "rows_per_strip", 279: "strip_bytes",
        282: "xres", 283: "yres", 284: "planar", 296: "res_unit", 305: "software",
        306: "datetime", 270: "description", 315: "artist", 339: "sample_format"}
TYPES = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 7: ("B", 1), 16: ("Q", 8)}


def read_tags(buf):
    bo = {b"II": "<", b"MM": ">"}[buf[:2]]
    assert struct.unpack(bo + "H", buf[2:4])[0] == 42, "not classic TIFF"
    ifd = struct.unpack(bo + "I", buf[4:8])[0]
    n = struct.unpack(bo + "H", buf[ifd:ifd + 2])[0]
    tags, raw = {}, {}
    for i in range(n):
        e = ifd + 2 + 12 * i
        tag, typ, cnt = struct.unpack(bo + "HHI", buf[e:e + 8])
        fmt, size = TYPES.get(typ, ("B", 1))
        total = size * cnt
        off = e + 8 if total <= 4 else struct.unpack(bo + "I", buf[e + 8:e + 12])[0]
        data = buf[off:off + total]
        if typ == 2:
            val = data.rstrip(b"\0").decode("latin1")
        elif typ == 5:
            v = struct.unpack(bo + "I" * (2 * cnt), data)
            val = [v[k] / v[k + 1] if v[k + 1] else 0 for k in range(0, len(v), 2)]
        else:
            val = list(struct.unpack(bo + fmt * cnt, data))
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        tags[TAGS.get(tag, tag)] = val
    nxt = struct.unpack(bo + "I", buf[ifd + 2 + 12 * n: ifd + 6 + 12 * n])[0]
    tags["_n_pages_gt1"] = nxt != 0
    return tags


def read_pixels(path):
    """Returns (tags, bytes) with pixels interleaved (chunky), row-major."""
    buf = open(path, "rb").read()
    t = read_tags(buf)
    assert t.get("compression", 1) == 1, f"compressed TIFF: {path}"
    offs, cnts = t["strip_offsets"], t["strip_bytes"]
    if not isinstance(offs, list):
        offs, cnts = [offs], [cnts]
    px = b"".join(buf[o:o + c] for o, c in zip(offs, cnts))
    return t, px
