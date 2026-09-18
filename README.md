# Calibração de Câmera

Trabalho 2 da disciplina de Visão Computacional (UFPR).

Dupla:
- Gustavo Gabriel Ripka - GRR20203935
- Arthur Barreto Godoi - GRR20224377

## Objetivo

Calibrar uma câmera a partir de fotografias de um tabuleiro quadriculado
(cada quadrado com 9 cm x 9 cm), obtendo a matriz de intrínsecos, os
coeficientes de distorção e a remoção da distorção das imagens. Também é
feito um experimento que, dada a posição de pontos no espaço 3D, determina
suas coordenadas nas imagens e compara com as observadas.

## Estrutura

```
cam-calibration/
  images/   fotos do tabuleiro usadas na calibração
  data/     resultados (matrizes, lista de imagens validas, erros)
  output/   figuras geradas (antes/depois da correcao, projecao)
  src/      scripts Python
  report/   relatorio em LaTeX
```

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/detect_pattern.py    # detecta o tamanho do tabuleiro
python src/calibrate.py         # calibra e salva data/calibration.npz
python src/undistort.py         # gera antes/depois em output/
python src/project_points.py    # experimento 3D -> 2D
```

O tabuleiro usado possui quadrados de 9 cm (`SQUARE_SIZE = 0.09` m nos
scripts).
