"""
Teste rapido: roda a identificacao de ganchos em UMA imagem local (sem precisar
de RTSP nem loop continuo). Util para validar que o modelo implantado no MVI
esta respondendo certo antes de rodar o monitor_hooks.py.

Uso:
    python inference/test_single_image.py --image caminho\para\imagem.png
"""
import argparse
import os

from dotenv import load_dotenv

from mvi_client import MVIClient
from monitor_hooks import annotate_frame, classify_hooks
import cv2


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Testa a identificacao em uma imagem")
    parser.add_argument("--image", required=True, help="Caminho da imagem a testar")
    parser.add_argument("--out", default="capturas/teste_anotado.jpg", help="Onde salvar a imagem anotada")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"Imagem nao encontrada: {args.image}")

    client = MVIClient.from_env()
    threshold = float(os.environ.get("MVI_CONFIDENCE_THRESHOLD", "0.5"))

    detections = client.infer_image(args.image, threshold=threshold)
    result = classify_hooks(detections)

    print(f"Total de ganchos detectados: {result['total_ganchos']}")
    print(f"Ocupados: {result['ocupados']}")
    print(f"Vazios: {result['vazios']}")
    for d in detections:
        print(f"  - {d.label} ({d.confidence:.2f}) em [{d.xmin:.0f},{d.ymin:.0f},{d.xmax:.0f},{d.ymax:.0f}]")

    frame = cv2.imread(args.image)
    annotated = annotate_frame(frame, detections)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, annotated)
    print(f"Imagem anotada salva em: {args.out}")


if __name__ == "__main__":
    main()
