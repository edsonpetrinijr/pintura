"""
Sincroniza a API de pecas (PartBldYJSON) com as cameras da linha: sempre que
aparece um registro NOVO de peca com gancho(s) atribuido(s), captura um frame
de cada camera e salva junto com os metadados (peca, ganchos, figura, etc.).

Isso gera automaticamente um dataset rotulado (imagem + qual peca esta em qual
gancho naquele instante), sem precisar desenhar caixas manualmente - depois da
para usar isso pra treinar o modelo (local ou no Maximo Visual Inspection).

Uso:
    python parts/sync_capture.py --poll-interval 5
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cameras import configured_cameras
from parts.parts_client import PartsClient, PartRecord

DEFAULT_API_URL = "http://b8wdwisep02.brazil.cat.com:3030/PartBldYJSON"


def default_cameras() -> dict:
    return configured_cameras()


def capture_snapshot(rtsp_url: str) -> "cv2.typing.MatLike | None":
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return None
    # o primeiro frame de um RTSP recem-aberto costuma vir do buffer do
    # decodificador: quadro incompleto, embacado ou de segundos atras. Como
    # aqui a imagem e casada com o instante do registro da API, um frame velho
    # rotula a peca errada. Descarta alguns antes de guardar.
    frame, ok = None, False
    for _ in range(10):
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()
    return frame if ok else None


def run(api_url: str, cameras: dict, poll_interval: float, out_dir: str, log_path: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    client = PartsClient(api_url)
    seen_keys: set[str] = set()

    log_is_new = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if log_is_new:
        writer.writerow(["timestamp_registro", "number_car", "part_number", "serial_number",
                          "hooks", "figure", "color", "program_robot", "camera", "imagem"])

    print(f"Sincronizando '{api_url}' com cameras: {list(cameras)}")

    try:
        while True:
            try:
                records = client.fetch()
            except Exception as exc:
                print(f"Erro ao consultar API de pecas: {exc}")
                time.sleep(poll_interval)
                continue

            for record in records:
                if record.key in seen_keys:
                    continue
                seen_keys.add(record.key)

                if not record.has_hooks:
                    continue

                handle_new_record(record, cameras, out_dir, writer, log_file)

            time.sleep(poll_interval)
    finally:
        log_file.close()


def handle_new_record(record: PartRecord, cameras: dict, out_dir: str, writer, log_file) -> None:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    hooks_str = ";".join(str(h) for h in record.hooks)
    print(f"Novo registro: peca {record.part_number} ({record.figure}) -> ganchos {hooks_str}")

    for camera_name, rtsp_url in cameras.items():
        frame = capture_snapshot(rtsp_url)
        if frame is None:
            print(f"  falha ao capturar frame de {camera_name} ({rtsp_url})")
            continue

        image_name = f"{ts}_{record.part_number}_{camera_name}.jpg"
        image_path = os.path.join(out_dir, image_name)
        cv2.imwrite(image_path, frame)

        writer.writerow([record.timestamp, record.number_car, record.part_number,
                          record.serial_number, hooks_str, record.figure, record.color,
                          record.program_robot, camera_name, image_name])
        log_file.flush()
        print(f"  salvo: {image_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Sincroniza API de pecas com as cameras da linha")
    parser.add_argument("--api-url", default=os.environ.get("PARTS_API_URL", DEFAULT_API_URL),
                         help="URL da API PartBldYJSON")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Segundos entre consultas a API")
    parser.add_argument("--out", default="dataset_labeled", help="Pasta de saida das imagens rotuladas")
    parser.add_argument("--log", default="logs/pecas_ganchos.csv", help="Arquivo CSV com os rotulos")
    args = parser.parse_args()

    cameras = default_cameras()
    if not cameras:
        raise SystemExit("Nenhuma camera configurada. Defina RTSP_URL/CAMERA_26_URL/CAMERA_27_URL no .env")

    run(args.api_url, cameras, args.poll_interval, args.out, args.log)


if __name__ == "__main__":
    main()
