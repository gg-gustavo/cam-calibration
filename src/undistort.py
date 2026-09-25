"""Remove a distorcao de imagens usando os parametros calibrados.

Gera em output/ comparacoes antes/depois para algumas imagens de exemplo.
"""

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "images")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")

N_SAMPLES = 3
THUMB_H = 520


def load_names():
    with open(os.path.join(DATA_DIR, "valid_images.txt")) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def thumb(img, h=THUMB_H):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * scale)), h))


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = np.load(os.path.join(DATA_DIR, "calibration.npz"))
    mtx = data["mtx"]
    dist = data["dist"]
    w, h = (int(v) for v in data["image_size"])

    names = load_names()
    idx = np.linspace(0, len(names) - 1, N_SAMPLES).astype(int)

    rows = []
    for i in idx:
        name = names[i]
        img = cv2.imread(os.path.join(IMAGES_DIR, name))
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        undist = cv2.undistort(img, mtx, dist, None, new_mtx)

        left = label(thumb(img), "original")
        right = label(thumb(undist), "sem distorcao")
        rows.append(np.hstack([left, right]))

        single = os.path.join(OUTPUT_DIR, f"undistort_{i:02d}.png")
        cv2.imwrite(single, undist)
        print(f"salvo {single}")

    grid = np.vstack(rows)
    out = os.path.join(OUTPUT_DIR, "distorcao_antes_depois.png")
    cv2.imwrite(out, grid)
    print(f"salvo {out}")

    with open(os.path.join(DATA_DIR, "undistort_info.txt"), "w") as f:
        f.write("Imagens usadas na comparacao antes/depois:\n")
        for i in idx:
            f.write(f"  {names[i]}\n")
        f.write("\nROI valida (x, y, w, h) = ")
        f.write(str(tuple(int(v) for v in roi)) + "\n")


if __name__ == "__main__":
    main()
