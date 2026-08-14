"""Rotulagem manual da ocupacao dos ganchos - o gabarito que hoje nao existe.

Por que isso e necessario: divergencia contra a API mistura erro do operador
com erro do nosso detector, e nao da para separar os dois sem alguem olhando a
imagem. Este script produz a verdade independente.

DUAS DECISOES DE PROJETO QUE IMPORTAM

1. Nao mostra o que o detector achou. Se mostrasse, quem rotula ancoraria na
   resposta da maquina e o resultado mediria concordancia, nao acuracia.

2. A ordem e ALEATORIA com semente fixa. Rotular em ordem cronologica faz a
   amostra virar "as primeiras N capturas do dia", que nao representa o dia.
   Com ordem aleatoria, parar no meio ainda deixa uma amostra valida.

Aceita "?" de proposito: forcar rotulo binario numa imagem duvidosa envenena o
gabarito, e gabarito ruim e pior que gabarito pequeno.

Teclas:
    o  ocupado          v  vazio           ?  nao da para dizer
    backspace  volta um gancho             n  pula a imagem inteira
    +/-  zoom            q ou ESC  sai (o progresso ja esta salvo)

Uso:
    .venv\\Scripts\\python.exe local_cv\\rotular_ganchos.py
    .venv\\Scripts\\python.exe local_cv\\rotular_ganchos.py --imagens dataset_validacao --amostra 50
"""
import argparse
import csv
import glob
import os
import random
import sys
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import load_hooks, pick_calibration
from cameras import CAMERA_NAMES

CAMPOS = ["imagem", "camera", "calibracao", "gancho", "ocupado", "rotulado_em"]
LARGURA = 1280
PAINEL = 420
CROP_W, CROP_ALTO, CROP_ACIMA = 120, 340, 50


def camera_do_nome(caminho: str) -> str | None:
    """dataset_validacao/20260805_090528_cabine.jpg -> cabine"""
    base = os.path.splitext(os.path.basename(caminho))[0]
    partes = set(base.split("_")) | set(base.split("-"))
    # Nome mais longo primeiro: "cabine2" contem "cabine" como substring e
    # casaria errado se testassemos na ordem alfabetica.
    for nome in sorted(CAMERA_NAMES, key=len, reverse=True):
        if nome in partes:
            return nome
    return None


def ja_rotuladas(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {l["imagem"] for l in csv.DictReader(f)}


def gravar(csv_path: str, linhas: list[dict]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    novo = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        w.writerows(linhas)


def recorte(frame, hook, zoom: float):
    """Faixa alta e estreita: na cabine o sinal e a CORRENTE descendo, nao o
    ponto do gancho. Recorte quadrado cortaria justamente a evidencia.

    zoom maior = recorte MENOR, que depois de ampliado para o painel aparece
    maior. Dividir em vez de multiplicar e o que faz o '+' aproximar.
    """
    alt, larg = frame.shape[:2]
    mw = max(12, int(CROP_W / zoom / 2))
    y0 = max(0, hook["y"] - int(CROP_ACIMA / zoom))
    y1 = min(alt, y0 + max(40, int(CROP_ALTO / zoom)))
    x0 = max(0, hook["x"] - mw)
    x1 = min(larg, hook["x"] + mw)
    corte = frame[y0:y1, x0:x1]
    if corte.size == 0:
        return np.zeros((CROP_ALTO, CROP_W, 3), np.uint8), (x0, y0)
    return corte, (x0, y0)


def montar(frame, hooks, indice, respostas, zoom, nome_img, feitas, total):
    escala = LARGURA / frame.shape[1]
    vista = cv2.resize(frame, (LARGURA, int(frame.shape[0] * escala)))

    for i, h in enumerate(hooks):
        px, py = int(h["x"] * escala), int(h["y"] * escala)
        marca = respostas.get(h["id"])
        cor = {"1": (0, 0, 255), "0": (0, 200, 0), "?": (0, 200, 255)}.get(marca, (170, 170, 170))
        if i == indice:
            cv2.circle(vista, (px, py), 16, (255, 255, 255), 2)
            cv2.circle(vista, (px, py), 20, (255, 0, 255), 2)
        cv2.circle(vista, (px, py), 7, cor, -1 if marca else 2)
        cv2.putText(vista, str(h["id"]), (px + 12, py - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    painel = np.zeros((vista.shape[0], PAINEL, 3), np.uint8)
    hook = hooks[indice]
    corte, _ = recorte(frame, hook, zoom)
    alvo_w = PAINEL - 20
    fator = min(alvo_w / corte.shape[1], (vista.shape[0] - 190) / max(corte.shape[0], 1))
    amp = cv2.resize(corte, (max(1, int(corte.shape[1] * fator)),
                             max(1, int(corte.shape[0] * fator))),
                     interpolation=cv2.INTER_CUBIC)
    ox = (PAINEL - amp.shape[1]) // 2
    painel[150:150 + amp.shape[0], ox:ox + amp.shape[1]] = amp
    cv2.rectangle(painel, (ox - 1, 149), (ox + amp.shape[1], 150 + amp.shape[0]),
                  (90, 90, 90), 1)

    marcados = sum(1 for h in hooks if respostas.get(h["id"]))
    texto = [
        (f"GANCHO {hook['id']}", 0.95, (255, 0, 255)),
        (f"{marcados}/{len(hooks)} neste frame", 0.5, (200, 200, 200)),
        (f"imagem {feitas + 1} de {total}", 0.5, (200, 200, 200)),
        (os.path.basename(nome_img)[:38], 0.42, (140, 140, 140)),
    ]
    y = 34
    for txt, tam, cor in texto:
        cv2.putText(painel, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, tam, cor, 1, cv2.LINE_AA)
        y += 30

    y = vista.shape[0] - 34
    for txt in ("q sai   n pula   backspace volta", "o ocupado   v vazio   ? duvida"):
        cv2.putText(painel, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (180, 180, 180), 1, cv2.LINE_AA)
        y -= 24

    return np.hstack([vista, painel])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagens", default="dataset_validacao")
    p.add_argument("--csv", default="logs/gabarito_ganchos.csv")
    p.add_argument("--camera", help="Rotula so as imagens desta camera")
    p.add_argument("--amostra", type=int, default=0, help="0 = todas")
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    caminhos = sorted(glob.glob(os.path.join(args.imagens, "**", "*.jpg"), recursive=True))
    if not caminhos:
        raise SystemExit(f"Nenhuma imagem em {args.imagens}/")

    feitas_antes = ja_rotuladas(args.csv)
    fila = []
    for c in caminhos:
        if c.replace("\\", "/") in feitas_antes or c in feitas_antes:
            continue
        cam = camera_do_nome(c)
        if cam and (not args.camera or cam == args.camera):
            fila.append((c, cam))

    if not fila:
        raise SystemExit(f"Nada novo para rotular ({len(feitas_antes)} imagens ja no gabarito).")

    random.Random(args.semente).shuffle(fila)
    if args.amostra:
        fila = fila[:args.amostra]

    print(f"{len(fila)} imagens na fila ({len(feitas_antes)} ja rotuladas antes).")
    print("o=ocupado  v=vazio  ?=duvida  backspace=volta  n=pula  q=sai\n")

    cv2.namedWindow("rotular", cv2.WINDOW_NORMAL)
    zoom, salvos = 1.0, 0

    for feitas, (caminho, cam) in enumerate(fila):
        frame = cv2.imread(caminho)
        if frame is None:
            print(f"ilegivel, pulando: {caminho}")
            continue
        try:
            calib, hooks, _ = pick_calibration(frame, cam, verbose=False)
        except Exception as exc:
            print(f"sem calibracao para {cam} ({exc}), pulando {caminho}")
            continue

        respostas: dict[int, str] = {}
        indice, sair, pular = 0, False, False

        while indice < len(hooks) and not sair and not pular:
            cv2.imshow("rotular", montar(frame, hooks, indice, respostas, zoom,
                                          caminho, feitas, len(fila)))
            tecla = cv2.waitKey(0) & 0xFF

            if tecla in (ord("q"), 27):
                sair = True
            elif tecla == ord("n"):
                pular = True
            elif tecla == 8:
                indice = max(0, indice - 1)
                respostas.pop(hooks[indice]["id"], None)
            elif tecla in (ord("+"), ord("=")):
                zoom = min(3.0, zoom + 0.25)
            elif tecla == ord("-"):
                zoom = max(0.5, zoom - 0.25)
            elif tecla in (ord("o"), ord("v"), ord("?"), ord("/")):
                respostas[hooks[indice]["id"]] = {ord("o"): "1", ord("v"): "0"}.get(tecla, "?")
                indice += 1

        if respostas and not sair:
            agora = datetime.now().isoformat(timespec="seconds")
            gravar(args.csv, [{
                "imagem": caminho.replace("\\", "/"),
                "camera": cam,
                "calibracao": os.path.basename(calib),
                "gancho": h["id"],
                "ocupado": respostas[h["id"]],
                "rotulado_em": agora,
            } for h in hooks if h["id"] in respostas])
            salvos += 1

        if sair:
            break

    cv2.destroyAllWindows()
    print(f"\n{salvos} imagens rotuladas nesta sessao -> {args.csv}")


if __name__ == "__main__":
    main()
