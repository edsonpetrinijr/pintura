"""
Calibracao por PONTO: clique uma vez em cima de cada gancho, na ordem em que
eles sao numerados pela API (gancho 1, 2, 3, ...). Salva um JSON com a posicao
de cada gancho para aquela camera.

ATENCAO A ORDEM: na cabine o gancho 1 e o do FUNDO (mais longe da camera, a
direita na imagem) e a numeracao cresce vindo em direcao a camera. Clicar da
esquerda para a direita inverte tudo, e o sintoma e a deteccao discordar da API
sem que nenhum ajuste de limiar melhore. Se acontecer, conserte com:

    python local_cv/renumber_hooks.py local_cv/hooks_cabine_11.json

Controles:
    clique esquerdo  -> marca o proximo gancho
    z                -> desfaz o ultimo ponto
    s / ENTER        -> salva e sai
    ESC / q          -> sai sem salvar

Uso:
    python local_cv/select_hook_points.py --image capturas/ref_cam26.jpg --out local_cv/hooks_cam26.json --scale 0.6
    python local_cv/select_hook_points.py --source rtsp://admin:2035@10.101.244.27:554 --out local_cv/hooks_cam27.json --scale 0.6
"""
import argparse
import json
import os

import cv2

WINDOW = "Clique em cada gancho na ordem (z=desfaz, s=salva, ESC=sai)"


def load_frame(image_path: str | None, source: str | None):
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise SystemExit(f"Nao consegui abrir a imagem: {image_path}")
        return frame

    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Nao consegui capturar frame de: {source}")
    return frame


def draw_points(canvas, points, scale: float):
    for i, (x, y) in enumerate(points, start=1):
        px, py = int(x * scale), int(y * scale)
        cv2.circle(canvas, (px, py), 6, (0, 255, 255), -1)
        cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1)
        cv2.putText(canvas, str(i), (px + 9, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Marca a posicao de cada gancho com um clique")
    parser.add_argument("--image", help="Imagem de referencia ja salva")
    parser.add_argument("--source", help="URL RTSP (usada se --image nao for informado)")
    parser.add_argument("--out", required=True, help="Arquivo JSON de saida com os pontos")
    parser.add_argument("--scale", type=float, default=0.6, help="Escala da janela de exibicao")
    parser.add_argument("--start-id", type=int, default=1, help="Numero do primeiro gancho")
    args = parser.parse_args()

    if not args.image and not args.source:
        raise SystemExit("Informe --image ou --source")

    frame = load_frame(args.image, args.source)
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x / args.scale), int(y / args.scale)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    print("Clique em cada gancho na ordem numerada pela API. z=desfaz, s=salva, ESC=sai.")

    while True:
        canvas = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        draw_points(canvas, points, args.scale)
        cv2.putText(canvas, f"{len(points)} ganchos marcados", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            print("Saiu sem salvar.")
            cv2.destroyAllWindows()
            return
        if key == ord("z") and points:
            points.pop()
        if key in (ord("s"), 13):
            break

    cv2.destroyAllWindows()

    hooks = [{"id": args.start_id + i, "x": x, "y": y} for i, (x, y) in enumerate(points)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"image_size": [frame.shape[1], frame.shape[0]], "hooks": hooks}, f, indent=2)

    print(f"{len(hooks)} ganchos salvos em {args.out}")


if __name__ == "__main__":
    main()
