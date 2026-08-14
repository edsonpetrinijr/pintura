"""Mede se a silhueta 3D separa a peca certa das erradas.

Nao e treino nem ajuste: so pontua cada frame rotulado contra TODOS os CAD
disponiveis e mostra onde a peca verdadeira ficou no ranking. Sem essa medida
nao existe limiar, e sem limiar "dar match" nao tem definicao.

Uso:
    .venv\\Scripts\\python.exe local_cv\\validar_match.py --camera cam26
"""
import argparse
import glob
import os
import re
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import SeletorDeCalibracao
from local_cv.identificar_peca import carregar_bancos
from local_cv.modelo3d import casar
from local_cv.silhueta import silhuetas

PADRAO = re.compile(r"^(\d{8}-\d{6})_(\d+)_(\w+)\.jpg$", re.IGNORECASE)


def frames_rotulados(pasta: str, camera: str, rotulos: set[str]) -> list[tuple[str, str]]:
    """(caminho, part number) dos frames desta camera cujo rotulo tem CAD."""
    achados = []
    for caminho in sorted(glob.glob(os.path.join(pasta, "*.jpg"))):
        m = PADRAO.match(os.path.basename(caminho))
        if not m:
            continue
        _, peca, cam = m.groups()
        if cam == camera and peca in rotulos:
            achados.append((caminho, peca))
    return achados


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset_labeled")
    p.add_argument("--camera", default="cam26")
    p.add_argument("--modelos", default="modelos")
    p.add_argument("--passo-yaw", type=int, default=15)
    args = p.parse_args()

    print(f"carregando CAD de {args.modelos}/ ...")
    bancos = carregar_bancos(args.modelos, args.passo_yaw, (-45, -30, -15, 0, 15, 30, 45))
    rotulos = set(bancos)
    print(f"{len(rotulos)} modelos: {', '.join(sorted(rotulos))}")

    casos = frames_rotulados(args.dataset, args.camera, rotulos)
    if not casos:
        raise SystemExit(f"Nenhum frame de {args.camera} com rotulo entre os CAD.")
    print(f"{len(casos)} frames rotulados em {args.camera}\n")

    seletor = SeletorDeCalibracao(args.camera)
    acertos = 0
    linhas = []
    certos, errados = [], []

    for caminho, verdade in casos:
        frame = cv2.imread(caminho)
        if frame is None:
            continue
        _, hooks, _, _ = seletor.escolher(frame)
        pecas = silhuetas(frame, hooks)
        if not pecas:
            linhas.append((os.path.basename(caminho), verdade, "-", 0.0, 0.0, "sem segmento"))
            continue

        # A peca rotulada e uma so, mas a camera ve varios ganchos. Uso o
        # segmento em que o modelo verdadeiro pontua melhor - qualquer outro
        # criterio penalizaria por um erro que nao e de identificacao.
        melhor_caso = None
        for peca in pecas:
            postos = casar(peca["mask"], bancos)
            por_modelo = {q["modelo"]: q["iou"] for q in postos}
            iou_certo = por_modelo.get(verdade, 0.0)
            if melhor_caso is None or iou_certo > melhor_caso[0]:
                melhor_caso = (iou_certo, postos, por_modelo)

        iou_certo, postos, por_modelo = melhor_caso
        top = postos[0]
        pior = max((v for k, v in por_modelo.items() if k != verdade), default=0.0)
        certos.append(iou_certo)
        errados.append(pior)
        ok = top["modelo"] == verdade
        acertos += ok
        linhas.append((os.path.basename(caminho), verdade, top["modelo"],
                       iou_certo, pior, "OK" if ok else "ERRO"))

    print(f"{'frame':<38} {'verdade':>8} {'top1':>8} {'iou_ok':>7} {'iou_err':>8}  res")
    for nome, verdade, top, a, b, res in linhas:
        print(f"{nome:<38} {verdade:>8} {top:>8} {a:7.3f} {b:8.3f}  {res}")

    n = len(linhas)
    print(f"\ntop-1 correto: {acertos}/{n} ({100*acertos/max(n,1):.0f}%)")
    if certos:
        c, e = np.array(certos), np.array(errados)
        print(f"iou da peca certa   media {c.mean():.3f}  min {c.min():.3f}  max {c.max():.3f}")
        print(f"iou do melhor errado media {e.mean():.3f}  min {e.min():.3f}  max {e.max():.3f}")
        print(f"margem (certo - errado) media {(c - e).mean():+.3f}  "
              f"positiva em {(c > e).sum()}/{len(c)}")


if __name__ == "__main__":
    main()
