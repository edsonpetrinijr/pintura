"""Compara ocupacao por SUBTRACAO DE FUNDO contra a de BORDAS, usando a API como
gabarito, offline, sobre logs/validacao_ganchos.csv.

So mede onde a API CONFIRMA ocupado (ocupado_api=1): esse e o unico rotulo
confiavel dela (peca sem programa de robo nunca aparece), e e o erro que custa -
perder uma peca faz o robo colidir. Recall = fracao desses ganchos que o metodo
pegou. Bordas ja esta no CSV (ocupado_local); a de fundo e calculada aqui.

Uso:
    python local_cv/occup_fundo_eval.py --background local_cv/background_cabine_median.jpg
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_cv.silhueta import mascara_fundo
from local_cv.detect_hooks_local import hook_window
from calibration import load_hooks


def cobertura(mask, janela) -> float:
    x1, y1, x2, y2 = janela
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / roi.size


def mascara_corrente(frame, fundo, limiar: int = 40) -> np.ndarray:
    """Subtracao de fundo SEM abertura: preserva a corrente fina do gancho.

    mascara_fundo abre com 7 px para tirar ruido, mas a corrente tem ~3-5 px e
    some junto. Aqui so um fechamento leve para ligar os elos.
    """
    dif = cv2.absdiff(frame, fundo).max(axis=2)
    bruta = (dif > limiar).astype(np.uint8) * 255
    return cv2.morphologyEx(bruta, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="logs/validacao_ganchos.csv")
    p.add_argument("--background", default="local_cv/background_cabine_median.jpg")
    p.add_argument("--win-w", type=int, default=40)
    p.add_argument("--win-h", type=int, default=220)
    p.add_argument("--drop-px", type=int, default=110,
                   help="quanto abaixo do gancho centrar a janela")
    args = p.parse_args()

    fundo = cv2.imread(args.background)
    if fundo is None:
        raise SystemExit(f"nao abriu o fundo: {args.background}")

    linhas = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    por_imagem = defaultdict(list)
    for r in linhas:
        if r["camera"] != "cabine":
            continue
        por_imagem[r["imagem"]].append(r)

    hooks_cache: dict[str, list] = {}
    frame_cache: dict[str, np.ndarray] = {}

    # carrega frames uma vez
    faltando = 0
    for imagem, rows in list(por_imagem.items()):
        if not os.path.exists(imagem):
            faltando += 1
            continue
        fr = cv2.imread(imagem)
        if fr is None or fr.shape[:2] != fundo.shape[:2]:
            faltando += 1
            continue
        frame_cache[imagem] = fr

    print(f"frames usados: {len(frame_cache)}  sem imagem: {faltando}")

    # baseline bordas (ja no CSV)
    todos = [r for rows in por_imagem.values() for r in rows
             if r["imagem"] in frame_cache]
    api1_rows = [r for r in todos if int(r["ocupado_api"]) == 1]
    api0_rows = [r for r in todos if int(r["ocupado_api"]) == 0]
    rec_bordas = np.mean([int(r["ocupado_local"]) for r in api1_rows])
    fp_bordas = np.mean([int(r["ocupado_local"]) for r in api0_rows])
    print(f"ganchos API-ocupado: {len(api1_rows)}   API-nao-lista: {len(api0_rows)}")
    print(f"\nBORDAS (atual):   recall(API=1) {rec_bordas:.2f}   "
          f"positivos(API=0) {fp_bordas:.2f}")

    limiares = [0.02, 0.04, 0.06, 0.10, 0.15]
    for nome, mask_fn in (("mascara_fundo (com abertura)", mascara_fundo),
                          ("mascara_corrente (sem abertura)", mascara_corrente)):
        mask_cache = {img: mask_fn(fr, fundo) for img, fr in frame_cache.items()}
        for drop in (40, 70, 110):
            reg1, reg0 = [], []
            for r in todos:
                calib = os.path.join("local_cv", r["calibracao"])
                if calib not in hooks_cache:
                    hooks_cache[calib] = load_hooks(calib)
                hooks = {h["id"]: h for h in hooks_cache[calib]}
                gid = int(r["gancho"])
                if gid not in hooks:
                    continue
                mask = mask_cache[r["imagem"]]
                jan = hook_window(hooks[gid], args.win_w, 0.0, drop,
                                  mask.shape, args.win_w, args.win_h)
                cov = cobertura(mask, jan)
                (reg1 if int(r["ocupado_api"]) == 1 else reg0).append(cov)
            print(f"\n{nome}  drop={drop}")
            print("  limiar  recall(API=1)  positivos(API=0)")
            for lim in limiares:
                rec = np.mean([1 if c > lim else 0 for c in reg1]) if reg1 else 0.0
                fp = np.mean([1 if c > lim else 0 for c in reg0]) if reg0 else 0.0
                print(f"  {lim:<7.2f} {rec:<14.2f} {fp:.2f}")

    print("\nrecall = pegou peca que a API confirma (alto = bom). meta: bater "
          "0.48 das bordas com positivos(API=0) parecido ou menor.")


if __name__ == "__main__":
    main()
