"""Write a PNG contact sheet (stdlib only). Usage: python -I contact_sheet.py <kaggle_3m dir> <out.png> case:slice ...
Columns per row: pre-contrast | FLAIR | post-contrast | FLAIR + mask outline (red)."""
import os, struct, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiffmini import read_pixels
W = 256

def png(path, w, h, rgb):
    raw = b"".join(b"\0" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw, 6)) + ch(b"IEND", b""))

root, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
tiles = []
for s in specs:
    case, sl = s.split(":")
    _, px = read_pixels(os.path.join(root, case, f"{case}_{sl}.tif"))
    _, m = read_pixels(os.path.join(root, case, f"{case}_{sl}_mask.tif"))
    row = []
    for k in range(3):
        g = px[k::3]; row.append(bytes(v for v in g for _ in range(3)))
    fl = bytearray(row[1])
    for i in range(W * W):
        if m[i]:
            y, x = divmod(i, W)
            if any(not (0 <= y + dy < W and 0 <= x + dx < W) or not m[(y + dy) * W + x + dx] for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0))):
                fl[3 * i:3 * i + 3] = b"\xff\x00\x00"
    row.append(bytes(fl)); tiles.append(row)
cols, rows = 4, len(tiles); w, h = cols * W, rows * W
img = bytearray(w * h * 3)
for r, row in enumerate(tiles):
    for c, t in enumerate(row):
        for y in range(W):
            o = ((r * W + y) * w + c * W) * 3
            img[o:o + W * 3] = t[y * W * 3:(y + 1) * W * 3]
png(out, w, h, bytes(img)); print(out)
