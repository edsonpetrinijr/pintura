"""
Identificacao local (sem Maximo/nuvem) de ganchos ocupados vs vazios, usando
apenas OpenCV classico. Duas formas de uso:

1) Com regioes calibradas (recomendado, mais confiavel):
   python local_cv/select_regions.py --image referencia.png   # marca 1x cada gancho
   python local_cv/detect_local.py --image frame.png --regions local_cv/regions.json

   Se voce tiver uma foto da linha com todos os ganchos VAZIOS, passe ela em
   --background para comparar por diferenca de pixels (mais preciso):
   python local_cv/detect_local.py --image frame.png --regions local_cv/regions.json --background linha_vazia.png

2) Modo automatico, sem calibrar nada (rapido, menos preciso) - so acha
   "blobs" (formas) na imagem toda via bordas/contornos:
   python local_cv/detect_local.py --image frame.png --auto
"""
import argparse
import json
import os

import cv2
import numpy as np


def detect_auto(frame, min_area: int, max_area: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        results.append({"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h, "area": area})
    return results


def classify_region(frame, region, background, diff_threshold: float, std_threshold: float):
    x1, y1, x2, y2 = region["xmin"], region["ymin"], region["xmax"], region["ymax"]
    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

    if background is not None:
        roi_bg = cv2.cvtColor(background[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(roi, roi_bg)
        score = float(np.mean(diff))
        occupied = score > diff_threshold
        return occupied, score

    score = float(np.std(roi))
    occupied = score > std_threshold
    return occupied, score


def run_regions(image_path: str, regions_path: str, background_path: str | None,
                 diff_threshold: float, std_threshold: float, out_path: str) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Nao consegui abrir a imagem: {image_path}")

    with open(regions_path, "r", encoding="utf-8") as f:
        regions = json.load(f)

    background = None
    if background_path:
        background = cv2.imread(background_path)
        if background is None:
            raise SystemExit(f"Nao consegui abrir a imagem de fundo: {background_path}")

    ocupados = 0
    vazios = 0
    for region in regions:
        occupied, score = classify_region(frame, region, background, diff_threshold, std_threshold)
        label = "OCUPADO" if occupied else "VAZIO"
        color = (0, 0, 255) if occupied else (0, 200, 0)
        ocupados += int(occupied)
        vazios += int(not occupied)

        cv2.rectangle(frame, (region["xmin"], region["ymin"]), (region["xmax"], region["ymax"]), color, 2)
        cv2.putText(frame, f"{region['id']}: {label} ({score:.1f})",
                    (region["xmin"], max(region["ymin"] - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"{region['id']}: {label} (score={score:.1f})")

    print(f"\nTotal: {len(regions)} | ocupados={ocupados} | vazios={vazios}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"Imagem anotada salva em: {out_path}")


def run_auto(image_path: str, min_area: int, max_area: int, out_path: str) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Nao consegui abrir a imagem: {image_path}")

    detections = detect_auto(frame, min_area, max_area)
    for d in detections:
        cv2.rectangle(frame, (d["xmin"], d["ymin"]), (d["xmax"], d["ymax"]), (0, 200, 255), 2)

    print(f"Modo automatico: {len(detections)} formas detectadas (sem classificar ocupado/vazio).")
    print("Ajuste --min-area/--max-area se detectar ruido demais ou de menos.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, frame)
    print(f"Imagem anotada salva em: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Identificacao local de ganchos (sem Maximo)")
    parser.add_argument("--image", required=True, help="Imagem/frame a analisar")
    parser.add_argument("--regions", help="Arquivo JSON com as regioes calibradas (select_regions.py)")
    parser.add_argument("--background", help="Imagem de referencia com os ganchos vazios (opcional)")
    parser.add_argument("--diff-threshold", type=float, default=15.0,
                         help="Diferenca media de pixel para considerar ocupado (com --background)")
    parser.add_argument("--std-threshold", type=float, default=25.0,
                         help="Desvio padrao de pixel para considerar ocupado (sem --background)")
    parser.add_argument("--auto", action="store_true", help="Modo automatico sem regioes calibradas")
    parser.add_argument("--min-area", type=int, default=500, help="Area minima de contorno (modo --auto)")
    parser.add_argument("--max-area", type=int, default=50000, help="Area maxima de contorno (modo --auto)")
    parser.add_argument("--out", default="capturas/teste_local.jpg", help="Onde salvar a imagem anotada")
    args = parser.parse_args()

    if args.auto:
        run_auto(args.image, args.min_area, args.max_area, args.out)
    else:
        if not args.regions:
            raise SystemExit("Informe --regions (gerado por select_regions.py) ou use --auto")
        run_regions(args.image, args.regions, args.background, args.diff_threshold, args.std_threshold, args.out)


if __name__ == "__main__":
    main()
