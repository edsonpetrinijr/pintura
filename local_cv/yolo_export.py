"""Gera um dataset YOLO-seg (segmentacao) a partir dos frames da cabine.

O rotulo de contorno sai de graca da subtracao de fundo que ja existe
(silhueta.mascara_fundo): a cabine e fixa, entao o que muda contra o fundo vazio
e exatamente a peca pendurada. Cada blob valido vira UM poligono de instancia no
formato YOLO-seg:

    <classe> x1 y1 x2 y2 ... xn yn        (todos normalizados 0..1)

NAO substitui revisao humana: a mascara erra em fumaca de tinta e operador
atravessando. Por isso o script tambem grava um overlay em <out>/preview/ para
conferir a olho quais frames prestam antes de treinar.

Classe: por padrao UMA classe "peca" (so contorno). Com --classe-por-nome ele le
o part number do nome do arquivo (AAAAMMDD-HHMMSS_<part>_<camera>.jpg) e usa como
classe - CORRETO so quando o carro carrega um tipo de peca so; com tipos
misturados no mesmo frame todos os blobs herdam o mesmo rotulo, que e errado.
Nesse caso deixe 1 classe e atribua o tipo na revisao.

Uso:
    python local_cv/yolo_export.py --frames dataset_validacao dataset_labeled \
        --background local_cv/background_cabine.jpg --out datasets/cabine_seg
"""
import argparse
import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_cv.silhueta import mascara_fundo, AREA_MIN

# So frames da cabine (.45). cabine2/cam26/cam27 nao tem fundo limpo que preste.
PADRAO_CABINE = "*_cabine.jpg"


def poligonos_do_frame(frame, fundo, area_min: int, epsilon_frac: float):
    """Um poligono simplificado por blob valido da mascara de subtracao de fundo."""
    mask = mascara_fundo(frame, fundo)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polis = []
    for c in contornos:
        if cv2.contourArea(c) < area_min:
            continue
        # simplifica: YOLO nao precisa de 400 vertices, e poligono enxuto treina
        # igual. epsilon proporcional ao perimetro mantem a forma.
        eps = epsilon_frac * cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(aprox) < 3:
            continue
        polis.append(aprox)
    return polis, mask


def classe_do_nome(caminho: str) -> str:
    partes = os.path.basename(caminho).split("_")
    return partes[1] if len(partes) >= 3 else "peca"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", nargs="+", required=True,
                   help="Pastas com frames da cabine")
    p.add_argument("--background", default="local_cv/background_cabine.jpg")
    p.add_argument("--out", required=True, help="Pasta do dataset YOLO a criar")
    p.add_argument("--classe-por-nome", action="store_true",
                   help="Usa o part number do nome como classe (1 tipo por carro)")
    p.add_argument("--area-min", type=int, default=AREA_MIN)
    p.add_argument("--epsilon", type=float, default=0.004,
                   help="Fracao do perimetro para simplificar o poligono")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    fundo = cv2.imread(args.background)
    if fundo is None:
        raise SystemExit(f"nao abriu o fundo: {args.background}")

    arquivos = []
    for pasta in args.frames:
        arquivos += glob.glob(os.path.join(pasta, PADRAO_CABINE))
    arquivos = sorted(set(arquivos))
    if not arquivos:
        raise SystemExit(f"nenhum frame {PADRAO_CABINE} em {args.frames}")

    # descobre as classes antes, para fixar os indices
    if args.classe_por_nome:
        nomes = sorted({classe_do_nome(a) for a in arquivos})
    else:
        nomes = ["peca"]
    idx_classe = {n: i for i, n in enumerate(nomes)}

    random.seed(args.seed)
    random.shuffle(arquivos)
    corte = int(len(arquivos) * (1 - args.val_frac))
    split = {a: ("train" if i < corte else "val") for i, a in enumerate(arquivos)}

    for sub in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", sub), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", sub), exist_ok=True)
    os.makedirs(os.path.join(args.out, "preview"), exist_ok=True)

    stats = {"train": 0, "val": 0, "vazios": 0, "instancias": 0}
    por_classe = {n: 0 for n in nomes}

    for caminho in arquivos:
        frame = cv2.imread(caminho)
        if frame is None or frame.shape[:2] != fundo.shape[:2]:
            continue
        sub = split[caminho]
        cls = idx_classe[classe_do_nome(caminho)] if args.classe_por_nome else 0
        cls_nome = nomes[cls]

        polis, mask = poligonos_do_frame(frame, fundo, args.area_min, args.epsilon)
        alt, larg = frame.shape[:2]

        base = os.path.splitext(os.path.basename(caminho))[0]
        img_dst = os.path.join(args.out, "images", sub, base + ".jpg")
        lbl_dst = os.path.join(args.out, "labels", sub, base + ".txt")
        shutil.copy(caminho, img_dst)

        linhas = []
        overlay = frame.copy()
        for poli in polis:
            xy = poli.astype(np.float32)
            xy[:, 0] /= larg
            xy[:, 1] /= alt
            coords = " ".join(f"{v:.6f}" for v in xy.flatten())
            linhas.append(f"{cls} {coords}")
            cv2.polylines(overlay, [poli], True, (0, 255, 0), 2)
            por_classe[cls_nome] += 1

        with open(lbl_dst, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))

        stats[sub] += 1
        stats["instancias"] += len(linhas)
        if not linhas:
            stats["vazios"] += 1
        cv2.imwrite(os.path.join(args.out, "preview", base + ".jpg"), overlay)

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        caminho_abs = os.path.abspath(args.out).replace("\\", "/")
        f.write(f"path: {caminho_abs}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(nomes)}\n")
        f.write("names: [" + ", ".join(f"'{n}'" for n in nomes) + "]\n")

    print(f"frames: {len(arquivos)}  train {stats['train']}  val {stats['val']}")
    print(f"instancias: {stats['instancias']}  frames sem peca (revisar): {stats['vazios']}")
    print("por classe:", {n: por_classe[n] for n in nomes})
    print(f"dataset em {args.out}  (confira {args.out}/preview antes de treinar)")


if __name__ == "__main__":
    main()
