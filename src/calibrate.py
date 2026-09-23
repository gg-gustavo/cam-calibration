"""Calibra a camera a partir das imagens validas detectadas.

Le data/valid_images.txt (gerado por detect_pattern.py), refina os cantos
com subpixel e roda cv2.calibrateCamera. Salva data/calibration.npz e um
resumo em data/calib_report.txt.
"""

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "images")
DATA_DIR = os.path.join(ROOT, "data")

SQUARE_SIZE = 0.09  # metros (9 cm)

FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FILTER_QUADS
)

SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

# Modelo de distorcao de 4 parametros (k1, k2, p1, p2). O termo k3 foi
# descartado: com ele o RMS praticamente nao melhora (1.098 vs 1.122 px) e
# a correcao passa a introduzir distorcao inversa (overfitting), porque as
# fotos foram tiradas a distancias parecidas e os coeficientes ficam mal
# condicionados.
DISTORTION_FLAGS = cv2.CALIB_FIX_K3


def load_valid():
    path = os.path.join(DATA_DIR, "valid_images.txt")
    names = []
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].strip().partition("=")
                meta[key.strip()] = value.strip()
                continue
            names.append(line)
    return names, meta


def main():
    names, meta = load_valid()
    cols, rows = (int(v) for v in meta["pattern_size"].split("x"))
    w, h = (int(v) for v in meta["image_size"].split("x"))
    pattern_size = (cols, rows)
    print(f"pattern {cols}x{rows}, image {w}x{h}, {len(names)} imagens")

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []
    used = []
    for name in names:
        path = os.path.join(IMAGES_DIR, name)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, FLAGS)
        if not ok:
            print(f"  ignorada (sem cantos): {name}")
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA
        )
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(name)

    print(f"{len(used)} imagens usadas na calibracao")

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None, flags=DISTORTION_FLAGS
    )

    per_image = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        diff = imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2)
        err = float(np.linalg.norm(diff, axis=1).mean())
        per_image.append(err)

    np.savez(
        os.path.join(DATA_DIR, "calibration.npz"),
        mtx=mtx,
        dist=dist,
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        image_size=np.array([w, h]),
        pattern_size=np.array(pattern_size),
        square_size=np.array(SQUARE_SIZE),
        rms=np.array(rms),
        names=np.array(used),
    )

    lines = []
    lines.append("Calibracao de camera")
    lines.append(f"pattern_size (cantos internos) = {cols} x {rows}")
    lines.append(f"image_size = {w} x {h}")
    lines.append(f"square_size = {SQUARE_SIZE} m")
    lines.append(f"imagens usadas = {len(used)}")
    lines.append("")
    lines.append(f"RMS (erro de reprojecao global) = {rms:.6f} px")
    lines.append("")
    lines.append("Matriz intrinseca K =")
    lines.append(np.array2string(mtx, precision=4, suppress_small=True))
    lines.append("")
    lines.append(f"fx = {mtx[0,0]:.4f} px")
    lines.append(f"fy = {mtx[1,1]:.4f} px")
    lines.append(f"cx = {mtx[0,2]:.4f} px")
    lines.append(f"cy = {mtx[1,2]:.4f} px")
    lines.append("")
    lines.append("Modelo de distorcao: 4 parametros (k3 fixado em 0)")
    lines.append("Coeficientes de distorcao [k1 k2 p1 p2] =")
    lines.append(np.array2string(dist.ravel(), precision=6, suppress_small=True))
    lines.append("")
    lines.append("Erro medio por imagem (px):")
    for name, err in zip(used, per_image):
        lines.append(f"  {err:8.4f}  {name}")
    lines.append("")
    lines.append(f"Erro medio = {np.mean(per_image):.4f} px")
    lines.append(f"Erro maximo = {np.max(per_image):.4f} px")

    report = "\n".join(lines)
    with open(os.path.join(DATA_DIR, "calib_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
