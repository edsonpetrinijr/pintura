"""Identifica a peca pendurada na cabine comparando com os modelos 3D.

Fluxo:
  1. segmenta a peca no frame (local_cv/silhueta.py)
  2. renderiza cada modelo 3D de muitos angulos (local_cv/modelo3d.py)
  3. escolhe o modelo cuja silhueta mais cobre a da camera

Nao ha treino nem dataset: a unica fonte de verdade sobre a forma e o CAD. Isso
resolve o gargalo real do projeto, que e ter so 2 a 6 fotos por tipo de peca, e
sem caixa delimitadora.

Coloque os arquivos em modelos/, um por peca, nomeados com o part number:

    modelos/6086032.step
    modelos/6460621-02-MOLDBOARD.stp

O nome ate o primeiro '-' vira o rotulo. STEP e convertido e cacheado sozinho.

Uso:
    python local_cv/identificar_peca.py --camera cabine
    python local_cv/identificar_peca.py --imagem capturas/frame.jpg
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import SeletorDeCalibracao
from cameras import CAMERA_NAMES, camera_url
from local_cv import modelo3d
from local_cv.modelo3d import banco_cacheado, casar, normalizar
from local_cv.silhueta import silhuetas

EXTENSOES = ("*.step", "*.stp", "*.glb", "*.stl", "*.obj")


def carregar_bancos(pasta: str, passo_yaw: int, pitches) -> dict[str, list[dict]]:
    caminhos = [c for e in EXTENSOES for c in glob.glob(os.path.join(pasta, e))]
    if not caminhos:
        raise SystemExit(
            f"Nenhum modelo 3D em {pasta}/. Coloque os arquivos .step com o "
            f"part number no nome (ex: {pasta}/6086032.step).")

    bancos = {}
    for caminho in sorted(caminhos):
        rotulo = os.path.splitext(os.path.basename(caminho))[0].split("-")[0]
        inicio = time.time()
        bancos[rotulo] = banco_cacheado(caminho, passo_yaw, pitches)
        gasto = time.time() - inicio
        print(f"  {rotulo}: {len(bancos[rotulo])} vistas"
              + (f" (renderizadas em {gasto:.0f}s)" if gasto > 2 else " (cache)"))
    return bancos


def frame_da_camera(camera: str):
    load_dotenv()
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(camera_url(camera))
    frame = None
    try:
        # As primeiras leituras vem do buffer e costumam estar velhas ou
        # corrompidas logo depois de abrir o stream.
        for _ in range(15):
            ok, novo = cap.read()
            if ok:
                frame = novo
    finally:
        cap.release()

    if frame is None:
        raise SystemExit(f"Nao consegui capturar de '{camera}'.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    fonte = parser.add_mutually_exclusive_group()
    fonte.add_argument("--camera", choices=CAMERA_NAMES, default="cabine")
    fonte.add_argument("--imagem", help="Analisa um arquivo em vez da camera")
    parser.add_argument("--modelos", default="modelos", help="Pasta com os CAD")
    parser.add_argument("--passo-yaw", type=int, default=10,
                         help="Passo do giro em torno do eixo vertical, em graus")
    parser.add_argument("--pitches", type=int, nargs="+", default=list(modelo3d.PITCHES),
                         help="Inclinacoes testadas, em graus")
    parser.add_argument("--top", type=int, default=3, help="Quantos candidatos mostrar")
    parser.add_argument("--gravar", help="Salva um comparativo visual neste caminho")
    args = parser.parse_args()

    print(f"modelos em {args.modelos}/:")
    bancos = carregar_bancos(args.modelos, args.passo_yaw, tuple(args.pitches))

    if args.imagem:
        frame = cv2.imread(args.imagem)
        if frame is None:
            raise SystemExit(f"Nao consegui ler {args.imagem}")
        camera = args.camera
    else:
        frame = frame_da_camera(args.camera)
        camera = args.camera

    _, hooks, _, encaixe = SeletorDeCalibracao(camera).escolher(frame)
    print(f"\nencaixe da calibracao: {100 * encaixe:.0f}%")

    achados = silhuetas(frame, hooks)
    if not achados:
        raise SystemExit("Nenhuma peca segmentada no frame.")

    print(f"{len(achados)} peca(s) segmentada(s)\n")
    comparativos = []
    for i, peca in enumerate(achados, 1):
        x, y, w, h = peca["caixa"]
        postos = casar(peca["mask"], bancos)
        melhor = postos[0]

        print(f"peca {i}: caixa=({x},{y}) {w}x{h}  razao={peca['razao']}")
        for p in postos[:args.top]:
            print(f"    {p['modelo']:>12s}  iou={p['iou']:.3f}  "
                  f"yaw={p['yaw']:.0f} pitch={p['pitch']:.0f}")
        # Sem folga sobre o segundo colocado a resposta nao vale: peca simetrica
        # ou parcialmente tapada casa igualmente bem com varios modelos.
        if melhor["vantagem"] is not None and melhor["vantagem"] < 0.05:
            print(f"    -> INCERTO: {melhor['modelo']} e o proximo empatam "
                  f"(vantagem {melhor['vantagem']:.3f})")
        else:
            print(f"    -> {melhor['modelo']}")

        if args.gravar:
            vista = next(v for v in bancos[melhor["modelo"]]
                         if v["yaw"] == melhor["yaw"] and v["pitch"] == melhor["pitch"])
            comparativos.append(np.hstack([normalizar(peca["mask"]), vista["mask"]]))

    if args.gravar and comparativos:
        os.makedirs(os.path.dirname(args.gravar) or ".", exist_ok=True)
        cv2.imwrite(args.gravar, np.vstack(comparativos))
        print(f"\ncomparativo salvo em {args.gravar}")


if __name__ == "__main__":
    main()
