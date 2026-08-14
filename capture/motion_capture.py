"""
Captura frames do RTSP so quando detecta movimentacao na faixa dos ganchos,
em vez de intervalo fixo. Usa subtracao de fundo (MOG2) restrita a uma janela
(ROI) que cobre a altura onde ficam os ganchos - assim ignora a esteira/corrente
de cima (que se move sempre) e as pessoas andando no chao mais embaixo.

A janela pode ser calculada automaticamente a partir das caixas marcadas em
local_cv/regions.json (select_regions.py), ou informada manualmente em pixels.

Uso (ROI automatica a partir das caixas ja marcadas nos ganchos):
    python capture/motion_capture.py --source rtsp://admin:2035@10.101.244.45:554 --out dataset_raw --cooldown 3 --regions local_cv/regions.json --margin 40

Uso (ROI manual em pixels):
    python capture/motion_capture.py --source rtsp://... --out dataset_raw --roi-y-min 200 --roi-y-max 550
"""
import argparse
import json
import os
import time
from datetime import datetime

import cv2


def roi_from_regions(regions_path: str, margin: int, frame_shape) -> tuple[int, int, int, int]:
    with open(regions_path, "r", encoding="utf-8") as f:
        regions = json.load(f)

    if not regions:
        raise SystemExit(f"'{regions_path}' nao tem nenhuma regiao marcada.")

    y_min = min(r["ymin"] for r in regions) - margin
    y_max = max(r["ymax"] for r in regions) + margin
    x_min = min(r["xmin"] for r in regions) - margin
    x_max = max(r["xmax"] for r in regions) + margin

    height, width = frame_shape[:2]
    return max(0, x_min), min(width, x_max), max(0, y_min), min(height, y_max)


def run(source: str, out_dir: str, min_area: int, cooldown: float, var_threshold: int,
        regions_path: str | None, margin: int,
        roi_x_min: int | None, roi_x_max: int | None,
        roi_y_min: int | None, roi_y_max: int | None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {source}")

    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=var_threshold, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (5, 5))

    last_saved = 0.0
    saved = 0
    x1 = x2 = y1 = y2 = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream encerrado ou frame invalido.")
                break

            if x1 is None:
                if regions_path:
                    x1, x2, y1, y2 = roi_from_regions(regions_path, margin, frame.shape)
                else:
                    height, width = frame.shape[:2]
                    x1 = roi_x_min if roi_x_min is not None else 0
                    x2 = roi_x_max if roi_x_max is not None else width
                    y1 = roi_y_min if roi_y_min is not None else 0
                    y2 = roi_y_max if roi_y_max is not None else height
                print(f"ROI de deteccao de movimento: x=[{x1},{x2}] y=[{y1},{y2}]")

            roi = frame[y1:y2, x1:x2]

            mask = subtractor.apply(roi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            motion_area = cv2.countNonZero(mask)

            if motion_area < min_area:
                continue

            now = time.monotonic()
            if now - last_saved < cooldown:
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
            filepath = os.path.join(out_dir, f"gancho-mov-{timestamp}.jpg")
            cv2.imwrite(filepath, frame)

            saved += 1
            last_saved = now
            print(f"[{saved}] movimento detectado (area={motion_area}) -> salvo: {filepath}")
    finally:
        cap.release()

    print(f"Concluido. {saved} frames salvos por movimento em '{out_dir}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura frames do RTSP por deteccao de movimento na faixa dos ganchos")
    parser.add_argument("--source", required=True, help="URL RTSP ou caminho de arquivo de video")
    parser.add_argument("--out", default="dataset_raw", help="Pasta de saida dos frames")
    parser.add_argument("--min-area", type=int, default=1500,
                         help="Quantidade minima de pixels em movimento (dentro da ROI) para disparar a captura")
    parser.add_argument("--cooldown", type=float, default=3.0,
                         help="Segundos minimos entre duas capturas (evita salvar muitas quase iguais)")
    parser.add_argument("--var-threshold", type=int, default=16,
                         help="Sensibilidade do MOG2 (menor = mais sensivel a pequenas mudancas)")
    parser.add_argument("--regions", default=None,
                         help="Caminho do regions.json (select_regions.py) para calcular a ROI automaticamente "
                              "a partir da altura onde os ganchos foram marcados")
    parser.add_argument("--margin", type=int, default=40,
                         help="Margem em pixels somada ao redor da ROI calculada a partir de --regions")
    parser.add_argument("--roi-x-min", type=int, default=None, help="ROI manual: x minimo em pixels")
    parser.add_argument("--roi-x-max", type=int, default=None, help="ROI manual: x maximo em pixels")
    parser.add_argument("--roi-y-min", type=int, default=None, help="ROI manual: y minimo em pixels")
    parser.add_argument("--roi-y-max", type=int, default=None, help="ROI manual: y maximo em pixels")
    args = parser.parse_args()

    run(args.source, args.out, args.min_area, args.cooldown, args.var_threshold,
        args.regions, args.margin, args.roi_x_min, args.roi_x_max, args.roi_y_min, args.roi_y_max)


if __name__ == "__main__":
    main()
