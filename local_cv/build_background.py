"""
Constroi uma imagem de fundo de referencia pela MEDIANA temporal de varios
frames da camera. Tudo que e fixo na cena (portas de vidro, monitores, paineis,
estrutura) sobrevive a mediana; pecas e correntes que passam sao removidas.

Esse fundo e usado por detect_hooks_local.py --background para decidir se um
gancho esta ocupado por diferenca de pixel, que e bem mais confiavel do que
densidade de bordas num cenario industrial cheio de estrutura.

Uso:
    python local_cv/build_background.py --camera cam26 --frames 40 --interval 15
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cameras import CAMERA_NAMES, camera_url


def build(rtsp_url: str, frames: int, interval: float, out_path: str) -> None:
    import time

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir: {rtsp_url}")

    collected = []
    try:
        while len(collected) < frames:
            ok, frame = cap.read()
            if not ok:
                print("Frame invalido, tentando de novo...")
                continue

            collected.append(frame)
            print(f"[{len(collected)}/{frames}] frame coletado")

            if len(collected) < frames:
                cap.release()
                time.sleep(interval)
                cap = cv2.VideoCapture(rtsp_url)
    finally:
        cap.release()

    stack = np.stack(collected, axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, background)
    print(f"\nFundo de referencia salvo em: {out_path}")


def de_pasta(padrao: str, out_path: str) -> None:
    """Mediana de frames que ja estao em disco.

    O coletor rotulado ja acumula frames dessas mesmas cameras fixas, entao dá
    para montar o fundo sem ocupar o stream RTSP - que e disputado: a .46 recusa
    conexao concorrente e a .45 fica presa quando outra ferramenta esta aberta.
    """
    arquivos = sorted(glob.glob(padrao))
    if not arquivos:
        raise SystemExit(f"Nenhum frame em {padrao}")

    pilha = np.stack([cv2.imread(a) for a in arquivos]).astype(np.float32)
    fundo = np.median(pilha, axis=0).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, fundo)
    print(f"{len(arquivos)} frames -> fundo salvo em: {out_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Constroi fundo de referencia por mediana temporal")
    parser.add_argument("--camera", default="cam26", choices=CAMERA_NAMES)
    parser.add_argument("--frames", type=int, default=40, help="Quantos frames coletar")
    parser.add_argument("--interval", type=float, default=15.0, help="Segundos entre frames")
    parser.add_argument("--pasta", help="monta de frames em disco em vez da camera, ex: dataset_labeled")
    parser.add_argument("--out", default=None, help="Padrao: local_cv/background_<camera>.jpg")
    args = parser.parse_args()

    out_path = args.out or f"local_cv/background_{args.camera}.jpg"
    if args.pasta:
        de_pasta(os.path.join(args.pasta, f"*_{args.camera}.jpg"), out_path)
    else:
        build(camera_url(args.camera), args.frames, args.interval, out_path)


if __name__ == "__main__":
    main()
