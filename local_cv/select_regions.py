"""
Ferramenta interativa para marcar a posicao de cada gancho em uma imagem de
referencia. Abre uma janela: arraste um retangulo por gancho, ENTER/ESPACO
para confirmar cada um, ESC para terminar.

Uso:
    python local_cv/select_regions.py --image ..\\vlcsnap-2026-07-06-09h53m13s474.png --out local_cv/regions.json
"""
import argparse
import json

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Marca as regioes dos ganchos em uma imagem")
    parser.add_argument("--image", required=True, help="Imagem de referencia (com a linha parada, se possivel)")
    parser.add_argument("--out", default="local_cv/regions.json", help="Onde salvar as regioes marcadas")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Nao consegui abrir a imagem: {args.image}")

    print("Arraste um retangulo por gancho. ENTER/ESPACO confirma cada um. ESC termina.")
    boxes = cv2.selectROIs("Marque cada gancho - ESC para terminar", frame, showCrosshair=True)
    cv2.destroyAllWindows()

    regions = []
    for i, (x, y, w, h) in enumerate(boxes):
        regions.append({
            "id": f"gancho_{i + 1}",
            "xmin": int(x),
            "ymin": int(y),
            "xmax": int(x + w),
            "ymax": int(y + h),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(regions, f, ensure_ascii=False, indent=2)

    print(f"{len(regions)} regioes salvas em {args.out}")


if __name__ == "__main__":
    main()
