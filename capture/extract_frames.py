"""
Extrai frames de um stream RTSP (ou arquivo de video) em intervalos regulares
para formar o dataset de treinamento do modelo de deteccao de ganchos.

Uso:
    python capture/extract_frames.py --source rtsp://.../stream1 --out dataset_raw --interval 5
    python capture/extract_frames.py --source video_gravado.mp4 --out dataset_raw --interval 2 --max-frames 500
"""
import argparse
import os
import time
from datetime import datetime

import cv2


def extract_frames(source: str, out_dir: str, interval: float, max_frames: int | None) -> int:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {source}")

    saved = 0
    last_saved_at = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            if now - last_saved_at < interval:
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
            filename = f"gancho-{timestamp}.jpg"
            filepath = os.path.join(out_dir, filename)
            cv2.imwrite(filepath, frame)

            saved += 1
            last_saved_at = now
            print(f"[{saved}] salvo: {filepath}")

            if max_frames is not None and saved >= max_frames:
                break
    finally:
        cap.release()

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai frames de RTSP/video para dataset")
    parser.add_argument("--source", required=True, help="URL RTSP ou caminho de arquivo de video")
    parser.add_argument("--out", default="dataset_raw", help="Pasta de saida dos frames")
    parser.add_argument("--interval", type=float, default=5.0, help="Segundos entre frames salvos")
    parser.add_argument("--max-frames", type=int, default=None, help="Numero maximo de frames a salvar")
    args = parser.parse_args()

    total = extract_frames(args.source, args.out, args.interval, args.max_frames)
    print(f"Concluido. {total} frames salvos em '{args.out}'.")


if __name__ == "__main__":
    main()
