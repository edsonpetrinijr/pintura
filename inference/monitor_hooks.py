"""
Monitora o stream RTSP da linha de pintura, captura frames periodicos,
envia para o modelo implantado no Maximo Visual Inspection e reporta quais
ganchos estao em uso (ocupados) e quais estao vazios.

Uso:
    python inference/monitor_hooks.py --interval 5

Convencao de rotulos esperada no dataset/modelo treinado no MVI:
    gancho_ocupado  -> gancho com peca pendurada
    gancho_vazio    -> gancho sem peca

Ajuste OCCUPIED_LABELS / EMPTY_LABELS abaixo se usar outros nomes de classe.
"""
import argparse
import csv
import os
import time
from datetime import datetime

import cv2
from dotenv import load_dotenv

from mvi_client import Detection, MVIClient

OCCUPIED_LABELS = {"gancho_ocupado", "gancho_com_peca"}
EMPTY_LABELS = {"gancho_vazio"}


def classify_hooks(detections: list[Detection]) -> dict:
    occupied = [d for d in detections if d.label in OCCUPIED_LABELS]
    empty = [d for d in detections if d.label in EMPTY_LABELS]
    return {
        "total_ganchos": len(occupied) + len(empty),
        "ocupados": len(occupied),
        "vazios": len(empty),
        "detections": detections,
    }


def annotate_frame(frame, detections: list[Detection]):
    for d in detections:
        color = (0, 0, 255) if d.label in OCCUPIED_LABELS else (0, 200, 0)
        cv2.rectangle(frame, (int(d.xmin), int(d.ymin)), (int(d.xmax), int(d.ymax)), color, 2)
        text = f"{d.label} {d.confidence:.2f}"
        cv2.putText(frame, text, (int(d.xmin), max(int(d.ymin) - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def run(source: str, interval: float, out_dir: str, log_path: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    client = MVIClient.from_env()
    threshold = float(os.environ.get("MVI_CONFIDENCE_THRESHOLD", "0.5"))

    log_is_new = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if log_is_new:
        writer.writerow(["timestamp", "total_ganchos", "ocupados", "vazios", "imagem"])

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {source}")

    last_check = 0.0
    tmp_path = os.path.join(out_dir, "_frame_tmp.jpg")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream encerrado ou frame invalido.")
                break

            now = time.monotonic()
            if now - last_check < interval:
                continue
            last_check = now

            cv2.imwrite(tmp_path, frame)

            try:
                detections = client.infer_image(tmp_path, threshold=threshold)
            except Exception as exc:  # falha de rede/API nao deve derrubar o loop
                print(f"Erro ao chamar MVI: {exc}")
                continue

            result = classify_hooks(detections)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            annotated = annotate_frame(frame.copy(), detections)
            image_name = f"gancho-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
            image_path = os.path.join(out_dir, image_name)
            cv2.imwrite(image_path, annotated)

            writer.writerow([timestamp, result["total_ganchos"], result["ocupados"],
                              result["vazios"], image_name])
            log_file.flush()

            print(f"[{timestamp}] ganchos={result['total_ganchos']} "
                  f"ocupados={result['ocupados']} vazios={result['vazios']}")
    finally:
        cap.release()
        log_file.close()


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Monitora ganchos via MVI")
    parser.add_argument("--source", default=os.environ.get("RTSP_URL"),
                         help="URL RTSP (padrao: variavel de ambiente RTSP_URL)")
    parser.add_argument("--interval", type=float, default=5.0, help="Segundos entre analises")
    parser.add_argument("--out", default="capturas", help="Pasta para frames anotados")
    parser.add_argument("--log", default="logs/ganchos.csv", help="Arquivo CSV de log")
    args = parser.parse_args()

    if not args.source:
        raise SystemExit("Informe --source ou defina RTSP_URL no .env")

    run(args.source, args.interval, args.out, args.log)


if __name__ == "__main__":
    main()
