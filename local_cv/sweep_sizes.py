"""
Varre combinacoes de tamanho de janela num frame ao vivo e mostra como o
conjunto de ganchos OCUPADOS muda. Serve para testar a hipotese de que a janela
escalonada com a perspectiva DILUI o score (densidade de bordas cai em janela
grande porque borda e 1-D e area e 2-D).

Captura UM frame da camera (uma stream so, respeita o limite de banda da .46),
salva em capturas/ e roda a varredura offline sobre esse mesmo frame.

Uso:
    python local_cv/sweep_sizes.py --camera cabine  --hooks local_cv/hooks_cabine_11.json
    python local_cv/sweep_sizes.py --camera cabine2 --hooks local_cv/hooks_cabine2_11.json
"""
import argparse
import os
import sys
import time
from datetime import datetime

import cv2
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import hook_confidence, load_hooks
from cameras import camera_url
from local_cv.detect_hooks_local import analyze
from parts.parts_client import PartsClient

DEFAULT_API_URL = "http://b8wdwisep02.brazil.cat.com:3030/PartBldYJSON"

# (min_size, max_size). min==max => janela FIXA (sem escalonar com perspectiva).
COMBOS = [
    (22, 110),   # default atual (escalonada)
    (40, 40),    # fixa 40
    (50, 50),    # fixa 50
    (60, 60),    # fixa 60
    (30, 60),    # escalonamento suave
    (40, 80),
]


def grab_frame(url: str, warmup: int = 8):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(f"nao abriu camera: {url}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.05)
    cap.release()
    if frame is None:
        raise SystemExit("nenhum frame lido")
    return frame


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--camera", required=True)
    p.add_argument("--hooks", required=True)
    p.add_argument("--threshold", type=float, default=4.5)
    p.add_argument("--factor", type=float, default=0.5)
    p.add_argument("--drop", type=float, default=0.8)
    p.add_argument("--api-url", default=os.environ.get("PARTS_API_URL", DEFAULT_API_URL))
    p.add_argument("--image", default=None, help="usa frame salvo em vez da camera")
    args = p.parse_args()

    hooks = load_hooks(args.hooks)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"nao leu imagem: {args.image}")
        origem = args.image
    else:
        frame = grab_frame(camera_url(args.camera))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        origem = f"capturas/sweep_{args.camera}_{stamp}.jpg"
        os.makedirs("capturas", exist_ok=True)
        cv2.imwrite(origem, frame)

    conf = hook_confidence(frame, hooks)
    print(f"camera={args.camera} frame={origem} conf_calib={conf:.1f} "
          f"ganchos={len(hooks)} threshold={args.threshold}")

    # API (rotulo confiavel so para OCUPADO)
    api_ocupados = None
    try:
        records = PartsClient(args.api_url).fetch()
        ocup = set()
        for r in records:
            ocup.update(r.hooks)
        api_ocupados = sorted(ocup)
        print(f"API ocupados={api_ocupados}  pecas={len(records)}")
    except Exception as exc:
        print(f"API indisponivel: {exc}")

    ids = [h["id"] for h in hooks]
    print()
    header = "min-max  ".ljust(10) + "".join(f"g{i:<5}" for i in ids) + " | ocupados"
    print(header)
    print("-" * len(header))

    for mn, mx in COMBOS:
        res = analyze(frame, hooks, None, args.threshold, args.factor,
                      mn, mx, args.drop, None)
        by_id = {r["id"]: r for r in res}
        row = f"{mn}-{mx}".ljust(10)
        for i in ids:
            row += f"{by_id[i]['score']:<6.1f}"
        occ = sorted(r["id"] for r in res if r["occupied"])
        row += f" | {occ}"
        print(row)

    print()
    print("scores por gancho acima; ocupado = score > threshold. "
          "Compare colunas: janela grande (default 22-110) deve baixar score "
          "dos ganchos da frente vs janela fixa.")


if __name__ == "__main__":
    main()
