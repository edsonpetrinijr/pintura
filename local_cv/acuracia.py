"""Mede a acuracia do NOSSO detector contra o gabarito humano.

Esta e a medida que separa as duas fontes de erro. A divergencia contra a API
mistura erro do operador com erro nosso; aqui a API nao entra. Rodamos o
detector nas imagens que alguem rotulou a mao e comparamos.

O resultado tem uma traducao que e o ponto central: com erro por gancho `p`, um
carro de N ganchos diverge por culpa nossa com probabilidade 1-(1-p)^N. Foi por
isso que medimos 100% de divergencia contra a API - basta p alto e N=11 para o
ruido do detector cobrir qualquer erro de operador.

Uso:
    .venv\\Scripts\\python.exe local_cv\\acuracia.py
    .venv\\Scripts\\python.exe local_cv\\acuracia.py --threshold 5.5
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import pick_calibration
from local_cv.detect_hooks_local import analyze


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gabarito", default="logs/gabarito_ganchos.csv")
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--factor", type=float, default=0.5)
    p.add_argument("--min-size", type=int, default=50)
    p.add_argument("--max-size", type=int, default=50)
    p.add_argument("--drop", type=float, default=0.8)
    args = p.parse_args()

    if not os.path.exists(args.gabarito):
        raise SystemExit(f"Nao achei {args.gabarito}. Rode local_cv/rotular_ganchos.py antes.")

    with open(args.gabarito, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    por_imagem = defaultdict(dict)
    for l in linhas:
        if l["ocupado"] in ("0", "1"):     # "?" fica de fora: nao e gabarito
            por_imagem[l["imagem"]][int(l["gancho"])] = l["ocupado"] == "1"

    duvidas = sum(1 for l in linhas if l["ocupado"] == "?")
    if not por_imagem:
        raise SystemExit("Gabarito sem nenhum rotulo decidido.")

    vp = vn = fp = fn = 0
    por_gancho = defaultdict(lambda: [0, 0])       # [erros, total]
    faltando = 0

    for caminho, verdade in sorted(por_imagem.items()):
        frame = cv2.imread(caminho)
        if frame is None:
            faltando += 1
            continue
        cam = next((l["camera"] for l in linhas if l["imagem"] == caminho), None)
        try:
            _, hooks, _ = pick_calibration(frame, cam, verbose=False)
        except Exception:
            faltando += 1
            continue

        for r in analyze(frame, hooks, None, args.threshold, args.factor,
                         args.min_size, args.max_size, args.drop, None, None, None):
            if r["id"] not in verdade:
                continue
            real, nosso = verdade[r["id"]], bool(r["occupied"])
            if real and nosso:
                vp += 1
            elif not real and not nosso:
                vn += 1
            elif nosso and not real:
                fp += 1
            else:
                fn += 1
            por_gancho[r["id"]][1] += 1
            por_gancho[r["id"]][0] += real != nosso

    n = vp + vn + fp + fn
    if not n:
        raise SystemExit("Nenhuma comparacao possivel (imagens sumiram?).")

    erro = (fp + fn) / n
    print(f"imagens no gabarito : {len(por_imagem)}"
          + (f"  ({faltando} ilegiveis/sem calibracao)" if faltando else ""))
    print(f"ganchos comparados  : {n}"
          + (f"  ({duvidas} marcados '?' ficaram de fora)" if duvidas else ""))
    print(f"limiar              : {args.threshold}\n")

    print(f"acerto por gancho   : {100*(vp+vn)/n:.1f}%   (erro {100*erro:.1f}%)")
    if vp + fn:
        print(f"  acha peca que existe    : {100*vp/(vp+fn):5.1f}%  ({vp}/{vp+fn})")
    if vn + fp:
        print(f"  reconhece gancho vazio  : {100*vn/(vn+fp):5.1f}%  ({vn}/{vn+fp})")
    print(f"  falso positivo {fp}   falso negativo {fn}\n")

    print("erro por gancho (concentracao aqui e geometria, nao operador):")
    for gancho, (erros, tot) in sorted(por_gancho.items()):
        barra = "#" * round(20 * erros / max(tot, 1))
        print(f"  gancho {gancho:>3d}: {erros:3d}/{tot:3d}  {100*erros/max(tot,1):5.1f}%  {barra}")

    print()
    for ganchos in (8, 11):
        prob = 1 - (1 - erro) ** ganchos
        print(f"projecao: carro de {ganchos} ganchos diverge por culpa NOSSA em "
              f"{100*prob:.0f}% dos casos")
    print("\nSe essa projecao ficar perto de 100%, a taxa medida contra a API e")
    print("nossa, nao do operador, e nenhum numero de carga sai dali.")


if __name__ == "__main__":
    main()
