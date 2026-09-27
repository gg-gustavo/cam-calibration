"""Experimento 3D -> 2D.

Dada a posicao de pontos no espaco 3D (os cantos do tabuleiro, com
quadrados de 9 cm), determina suas coordenadas em algumas imagens e
confere com os cantos observados.

Faz duas verificacoes:
  A) projecao direta com os parametros calibrados;
  B) leave-one-out: calibra sem a imagem de teste e projeta nela, para
     conferir que a calibracao prediz coordenadas em imagens novas.
"""

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "images")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")

SQUARE_SIZE = 0.09
N_TEST = 3
THUMB_H = 520

FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FILTER_QUADS
)
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def load_meta():
    meta = {}
    names = []
    with open(os.path.join(DATA_DIR, "valid_images.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition("=")
                meta[k.strip()] = v.strip()
            else:
                names.append(line)
    cols, rows = (int(v) for v in meta["pattern_size"].split("x"))
    return names, (cols, rows)


def board_object_points(cols, rows):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * SQUARE_SIZE


def detect(path, pattern_size):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, FLAGS)
    if not ok:
        return None, gray
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX)
    return corners, gray


def calibrate(objpoints, imgpoints, size):
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, size, None, None, flags=cv2.CALIB_FIX_K3
    )
    return mtx, dist


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names, (cols, rows) = load_meta()
    objp = board_object_points(cols, rows)

    objpoints = []
    imgpoints = []
    for name in names:
        c, _ = detect(os.path.join(IMAGES_DIR, name), (cols, rows))
        objpoints.append(objp)
        imgpoints.append(c)

    data = np.load(os.path.join(DATA_DIR, "calibration.npz"))
    mtx = data["mtx"]
    dist = data["dist"]
    w, h = (int(v) for v in data["image_size"])

    test_idx = np.linspace(0, len(names) - 1, N_TEST).astype(int)

    lines = []
    lines.append("Experimento 3D -> 2D")
    lines.append(f"Pontos 3D = {cols*rows} cantos do tabuleiro (quadrado 9 cm)")
    lines.append(f"image_size = {w}x{h}")
    lines.append("")

    thumbs = []
    for i in test_idx:
        name = names[i]
        corners, gray = detect(os.path.join(IMAGES_DIR, name), (cols, rows))

        ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        projA, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        errA = np.linalg.norm(
            corners.reshape(-1, 2) - projA.reshape(-1, 2), axis=1
        )

        train_o = [objpoints[j] for j in range(len(names)) if j != i]
        train_i = [imgpoints[j] for j in range(len(names)) if j != i]
        mtx_lo, dist_lo = calibrate(train_o, train_i, (w, h))
        ok, rvec2, tvec2 = cv2.solvePnP(objp, corners, mtx_lo, dist_lo)
        projB, _ = cv2.projectPoints(objp, rvec2, tvec2, mtx_lo, dist_lo)
        errB = np.linalg.norm(
            corners.reshape(-1, 2) - projB.reshape(-1, 2), axis=1
        )

        lines.append(f"Imagem: {name}")
        lines.append(
            f"  (A) projecao direta      erro medio {errA.mean():.3f} px"
            f"  max {errA.max():.3f} px"
        )
        lines.append(
            f"  (B) leave-one-out        erro medio {errB.mean():.3f} px"
            f"  max {errB.max():.3f} px"
        )
        lines.append("")

        scale = THUMB_H / h
        vis = cv2.resize(gray, (int(round(w * scale)), THUMB_H))
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        for (x, y) in corners.reshape(-1, 2):
            cv2.circle(vis, (int(x * scale), int(y * scale)), 6, (0, 255, 0), 2)
        for (x, y) in projB.reshape(-1, 2):
            px, py = int(x * scale), int(y * scale)
            cv2.drawMarker(vis, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            vis,
            f"LOO err {errB.mean():.2f} px",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        thumbs.append(vis)

    grid = np.hstack(thumbs)
    out = os.path.join(OUTPUT_DIR, "projecao_3d2d.png")
    cv2.imwrite(out, grid)
    lines.append("Verde = cantos observados; vermelho = pontos 3D projetados.")
    lines.append(f"Figura: {os.path.relpath(out, ROOT)}")

    report = "\n".join(lines)
    with open(os.path.join(DATA_DIR, "projection_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
