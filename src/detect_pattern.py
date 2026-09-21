"""Detecta o tamanho do tabuleiro e as imagens validas para calibracao.

Percorre todas as imagens em images/ testando varios tamanhos candidatos de
tabuleiro (numero de cantos internos por linha x coluna) e escolhe o tamanho
que e detectado no maior numero de imagens. Salva a lista de imagens validas
em data/valid_images.txt.
"""

import glob
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "images")
DATA_DIR = os.path.join(ROOT, "data")

FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FILTER_QUADS
)

CANDIDATES = [
    (c, r)
    for c in range(4, 12)
    for r in range(3, 11)
    if c > r
]


def detect(img_gray, pattern_size):
    ok, corners = cv2.findChessboardCorners(img_gray, pattern_size, FLAGS)
    return ok, corners


def main():
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")))
    print(f"{len(images)} imagens encontradas")

    gray_cache = {}
    for path in images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        gray_cache[path] = img

    results = {}
    for pattern_size in CANDIDATES:
        count = 0
        for path in images:
            ok, _ = detect(gray_cache[path], pattern_size)
            if ok:
                count += 1
        results[pattern_size] = count
        print(f"  cantos {pattern_size[0]}x{pattern_size[1]}: {count} imagens")

    best = max(results, key=lambda k: results[k])
    print(f"\nMelhor tamanho: {best[0]}x{best[1]} ({results[best]} imagens)")

    valid = [p for p in images if detect(gray_cache[p], best)[0]]

    sizes = {}
    for p in valid:
        h, w = gray_cache[p].shape
        sizes.setdefault((w, h), []).append(p)
    for size, group in sorted(sizes.items(), key=lambda kv: -len(kv[1])):
        print(f"  resolucao {size[0]}x{size[1]}: {len(group)} imagens validas")

    main_size = max(sizes, key=lambda k: len(sizes[k]))
    dropped = [p for p in valid if p not in sizes[main_size]]
    valid = sizes[main_size]
    print(f"Resolucao usada: {main_size[0]}x{main_size[1]} ({len(valid)} imagens)")
    if dropped:
        print(f"Descartadas por resolucao diferente: {len(dropped)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "valid_images.txt")
    with open(out, "w") as f:
        f.write(f"# pattern_size={best[0]}x{best[1]}\n")
        f.write(f"# image_size={main_size[0]}x{main_size[1]}\n")
        for p in valid:
            f.write(os.path.basename(p) + "\n")
    print(f"{len(valid)} imagens validas salvas em {out}")


if __name__ == "__main__":
    main()
