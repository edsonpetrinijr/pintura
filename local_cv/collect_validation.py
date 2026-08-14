"""
Coleta continua para validar e evoluir a deteccao de ganchos.

A cada intervalo (1 minuto por padrao):
  1. captura uma rajada de frames da cabine e mede o score de cada gancho pela
     mediana da rajada, para nao registrar um frame ruim isolado
  2. escolhe a calibracao pela imagem e classifica cada gancho
  3. le o estado atual da API de pecas, que serve de gabarito
  4. grava uma linha por gancho no CSV e guarda as imagens

O CSV e a materia-prima do learn_thresholds.py, que ajusta o limiar de cada
gancho a partir desses dados. Por isso ele grava o rotulo da API gancho a
gancho, e nao so a lista.

IMPORTANTE - api_ok: quando a API nao responde, a lista de ganchos ocupados
volta vazia, o que e indistinguivel de "carro sem nada pendurado". Gravar isso
como gabarito ensinaria a deteccao errado. A coluna api_ok marca essas linhas
para que o aprendizado as descarte.

Uso:
    python local_cv/collect_validation.py
    python local_cv/collect_validation.py --interval 20 --somente-mudancas
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import EncaixeDaCalibracao, load_thresholds, pick_calibration
from cameras import configured_cameras
from local_cv.detect_hooks_local import analyze
from local_cv.stability import BAIXA, HookStabilizer
from parts.parts_client import PartsClient

DEFAULT_API_URL = "http://b8wdwisep02.brazil.cat.com:3030/PartBldYJSON"

CSV_FIELDS = [
    "timestamp", "camera", "calibracao", "confianca_calib",
    "gancho", "score", "score_mediana", "ocupado_local", "certeza",
    "api_ok", "ocupado_api", "api_ganchos_ocupados", "api_pecas",
    "number_car", "imagem",
]


def snapshot(rtsp_url: str, quantos: int = 1, espaco: float = 0.4):
    """Captura `quantos` frames espacados, para medir pela mediana da rajada.

    O timeout do ffmpeg nao e detalhe: sem ele uma camera fora do ar segura o
    VideoCapture por dezenas de segundos, e numa coleta de um dia inteiro isso
    trava o ciclo em vez de registrar a falha e seguir.
    """
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                          "rtsp_transport;tcp|stimeout;5000000")
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frames = []
    for i in range(quantos):
        if i:
            time.sleep(espaco)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def estado_local(frames, camera: str, threshold: float, factor: float,
                  min_size: int, max_size: int, drop: float, margem: float):
    """Classifica os ganchos usando a rajada inteira, nao um frame so."""
    caminho, hooks, confianca = pick_calibration(frames[0], camera, verbose=False)
    limiares = load_thresholds(caminho)

    estabilizador = HookStabilizer(len(frames), margem)
    results = []
    for frame in frames:
        brutos = analyze(frame, hooks, None, threshold, factor, min_size, max_size,
                          drop, None, None, None)
        results = estabilizador.aplicar(brutos, threshold, limiares)

    return os.path.basename(caminho), confianca, results


def append_rows(csv_path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            cabecalho = next(csv.reader(f), [])
        if cabecalho and cabecalho != CSV_FIELDS:
            # Formato antigo: anexar aqui embaralharia as colunas. Guarda o
            # arquivo velho em vez de perder ou corromper o que ja foi coletado.
            antigo = f"{os.path.splitext(csv_path)[0]}_ate_{datetime.now():%Y%m%d_%H%M%S}.csv"
            os.rename(csv_path, antigo)
            print(f"CSV com colunas antigas movido para {antigo}; comecando um novo.")

    novo = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if novo:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Coleta continua para validar a deteccao de ganchos")
    parser.add_argument("--camera", nargs="+", default=["cabine"],
                         help="Cameras analisadas. Passe mais de uma para cruzar vistas "
                              "do mesmo gancho, ex: --camera cabine cabine2")
    parser.add_argument("--interval", type=float, default=60.0, help="Segundos entre capturas")
    parser.add_argument("--rajada", type=int, default=5,
                         help="Frames por captura; o score e a mediana deles")
    parser.add_argument("--csv", default="logs/validacao_ganchos.csv")
    parser.add_argument("--image-dir", default="dataset_validacao")
    parser.add_argument("--somente-mudancas", action="store_true",
                         help="Salva imagem so quando a ocupacao muda (padrao: toda captura)")
    parser.add_argument("--api-url", default=os.environ.get("PARTS_API_URL", DEFAULT_API_URL))
    parser.add_argument("--threshold", type=float, default=4.5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=50)
    parser.add_argument("--drop", type=float, default=0.8)
    parser.add_argument("--margem", type=float, default=1.0,
                         help="Zona morta da histerese, igual a da tela")
    parser.add_argument("--encaixe-min", type=float, default=0.7,
                         help="Fracao do pico recente da calibracao abaixo da qual a "
                              "cena esta em transicao e a leitura nao vale. A medida "
                              "bruta muda com carro, luz e fumaca, entao limiar "
                              "absoluto marca cena boa como ruim")
    args = parser.parse_args()

    # Um por camera: cada vista tem o seu proprio nivel tipico de encaixe.
    # Meia-vida longa porque aqui a amostragem e a cada --interval, nao por frame.
    encaixes = {nome: EncaixeDaCalibracao(meia_vida=3600.0, fracao=args.encaixe_min)
                for nome in args.camera}

    cameras = configured_cameras()
    faltando = [c for c in args.camera if c not in cameras]
    if faltando:
        raise SystemExit(f"Cameras nao configuradas no .env: {faltando}. Tem: {sorted(cameras)}")

    client = PartsClient(args.api_url)
    os.makedirs(args.image_dir, exist_ok=True)

    print(f"Coletando a cada {args.interval:.0f}s (rajada de {args.rajada} frames). "
          f"Analisando {args.camera}, guardando imagens de {sorted(cameras)}. "
          f"Ctrl+C para parar.")
    print(f"CSV: {args.csv}")
    if "cabine" in cameras:
        print("AVISO: a .45 aceita um stream por vez. Nao deixe o monitor ou a "
              "sobreposicao abertos junto, ou uma das duas vai receber frame velho.")
    print()

    ultimo_estado = {}
    total = 0

    while True:
        try:
            agora = datetime.now()
            ts = agora.strftime("%Y%m%d_%H%M%S")

            # Captura tudo antes de analisar: as vistas precisam ser do mesmo
            # instante para poderem ser cruzadas depois.
            capturas = {nome: snapshot(cameras[nome], args.rajada) for nome in args.camera}
            if not any(capturas.values()):
                print(f"[{ts}] nenhuma camera respondeu, pulando")
                time.sleep(args.interval)
                continue

            try:
                records = client.fetch()
                api_ok = True
            except requests.RequestException as exc:
                print(f"[{ts}] API indisponivel ({exc}); linhas marcadas api_ok=0")
                records, api_ok = [], False

            api_ocupados = sorted({h for r in records for h in r.hooks})
            api_pecas = ";".join(sorted({r.part_number for r in records}))
            # Sem o carro nao da para medir taxa de erro: um carro parado dez
            # minutos vira dez divergencias no CSV e a taxa sai inflada.
            carros = ";".join(sorted({str(r.number_car) for r in records
                                      if r.number_car is not None}))

            leituras = {}
            for nome in args.camera:
                frames = capturas.get(nome)
                if not frames:
                    print(f"[{ts}] falha ao capturar {nome}")
                    continue

                calib, confianca, results = estado_local(
                    frames, nome, args.threshold, args.factor,
                    args.min_size, args.max_size, args.drop, args.margem)

                # Encaixe baixo = carro entrando/saindo, pontos fora dos
                # ganchos. A linha continua sendo gravada (confianca_calib fica
                # no CSV e o aprendizado filtra por ela), mas nao vale como
                # leitura.
                if not encaixes[nome].confiavel(encaixes[nome].avaliar(confianca)):
                    results = [{**r, "certeza": BAIXA} for r in results]

                leituras[nome] = (calib, confianca, results)

            estado = {n: tuple(r["occupied"] for r in v[2]) for n, v in leituras.items()}
            mudou = estado != ultimo_estado
            ultimo_estado = estado

            imagens = {}
            if mudou or not args.somente_mudancas:
                for nome, url in cameras.items():
                    quadros = capturas.get(nome) or snapshot(url)
                    if not quadros:
                        continue
                    caminho = os.path.join(args.image_dir, f"{ts}_{nome}.jpg")
                    cv2.imwrite(caminho, quadros[-1])
                    imagens[nome] = caminho

            for nome, (calib, confianca, results) in leituras.items():
                append_rows(args.csv, [{
                    "timestamp": agora.isoformat(timespec="seconds"),
                    "camera": nome,
                    "calibracao": calib,
                    "confianca_calib": round(confianca, 1),
                    "gancho": r["id"],
                    "score": round(r["score_bruto"], 2),
                    "score_mediana": round(r["score"], 2),
                    "ocupado_local": int(r["occupied"]),
                    "certeza": r["certeza"],
                    "api_ok": int(api_ok),
                    "ocupado_api": int(r["id"] in api_ocupados) if api_ok else "",
                    "api_ganchos_ocupados": ";".join(str(h) for h in api_ocupados),
                    "api_pecas": api_pecas,
                    "number_car": carros,
                    "imagem": imagens.get(nome, ""),
                } for r in results])

                ocupados = [r["id"] for r in results if r["occupied"]]
                incertos = [r["id"] for r in results if r["certeza"] == BAIXA]
                print(f"[{ts}]{'*' if mudou else ' '} {nome:8s} {calib} conf={confianca:4.1f} | "
                      f"ocupados: {ocupados or 'nenhum'} | "
                      f"incertos: {incertos or 'nenhum'}")

            print(f"{'':19s} API: {api_ocupados if api_ok else 'indisponivel'}"
                  f"{'  (imagens salvas)' if imagens else ''}")
            total += 1

        except KeyboardInterrupt:
            print(f"\nParado. {total} capturas gravadas em {args.csv}")
            return
        except Exception as exc:
            print(f"Erro no ciclo: {exc}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
