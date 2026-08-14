"""
Visualizacao: pega o estado atual da API de pecas + um frame da camera e desenha
    - um pontinho em cima de CADA gancho (verde = vazio, vermelho = ocupado)
    - uma caixa em volta de cada peca, cobrindo os ganchos que ela ocupa
    - o rotulo da peca (part_number / figure) ao lado da caixa

Requer a calibracao de pontos feita com local_cv/select_hook_points.py.

Uso (snapshot unico):
    python parts/visualize.py --camera cam26 --out capturas/overlay_cam26.jpg

Uso (ao vivo, atualizando):
    python parts/visualize.py --camera cam26 --watch --interval 5
"""
import argparse
import os
import sys
from datetime import datetime

import cv2
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import pick_calibration
from cameras import CAMERA_NAMES, camera_url
from parts.parts_client import PartsClient, PartRecord

DEFAULT_API_URL = "http://b8wdwisep02.brazil.cat.com:3030/PartBldYJSON"

COLOR_EMPTY = (0, 200, 0)
COLOR_OCCUPIED = (0, 0, 255)
COLOR_BOX = (0, 200, 255)


def current_occupancy(records: list[PartRecord]) -> dict[int, PartRecord]:
    """Mapeia gancho -> peca mais recente que o ocupa."""
    occupancy: dict[int, PartRecord] = {}
    for record in sorted(records, key=lambda r: r.timestamp):
        for hook_id in record.hooks:
            occupancy[hook_id] = record
    return occupancy


def capture_frame(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Nao consegui capturar frame de: {rtsp_url}")
    return frame


def draw_overlay(frame, hooks: list[dict], occupancy: dict[int, PartRecord],
                  pad_x: int, part_height: int):
    hook_by_id = {h["id"]: h for h in hooks}

    # 1) caixa por peca (agrupa os ganchos que a mesma peca ocupa)
    parts_to_hooks: dict[str, list[int]] = {}
    for hook_id, record in occupancy.items():
        if hook_id in hook_by_id:
            parts_to_hooks.setdefault(record.key, []).append(hook_id)

    for part_key, hook_ids in parts_to_hooks.items():
        record = occupancy[hook_ids[0]]
        xs = [hook_by_id[h]["x"] for h in hook_ids]
        ys = [hook_by_id[h]["y"] for h in hook_ids]

        x1 = max(0, min(xs) - pad_x)
        x2 = min(frame.shape[1], max(xs) + pad_x)
        y1 = max(0, min(ys) - 15)
        y2 = min(frame.shape[0], max(ys) + part_height)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 3)

        label = f"{record.part_number}"
        if record.figure:
            label += f" ({record.figure})"
        label += f" - ganchos {','.join(str(h) for h in sorted(hook_ids))}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), COLOR_BOX, -1)
        cv2.putText(frame, label, (x1 + 4, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 2) pontinho em cada gancho
    for hook in hooks:
        occupied = hook["id"] in occupancy
        color = COLOR_OCCUPIED if occupied else COLOR_EMPTY
        center = (hook["x"], hook["y"])

        cv2.circle(frame, center, 9, (0, 0, 0), -1)
        cv2.circle(frame, center, 7, color, -1)
        cv2.putText(frame, str(hook["id"]), (hook["x"] + 11, hook["y"] - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3) resumo no topo
    total = len(hooks)
    ocupados = sum(1 for h in hooks if h["id"] in occupancy)
    resumo = f"{datetime.now():%Y-%m-%d %H:%M:%S}  |  ganchos: {total}  ocupados: {ocupados}  vazios: {total - ocupados}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, resumo, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


def render_once(client: PartsClient, rtsp_url: str, camera: str,
                 hooks_arg: str | None, car_hooks: int | None,
                 out_path: str, pad_x: int, part_height: int) -> None:
    records = client.fetch()
    occupancy = current_occupancy(records)
    frame = capture_frame(rtsp_url)

    # A escolha e por frame: o carro na linha muda, e com ele a quantidade de ganchos.
    _, hooks, _ = pick_calibration(frame, camera, hooks_arg, car_hooks)

    frame = draw_overlay(frame, hooks, occupancy, pad_x, part_height)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, frame)

    ocupados = sorted(h["id"] for h in hooks if h["id"] in occupancy)
    print(f"ocupados: {ocupados or 'nenhum'} -> {out_path}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Desenha ganchos e pecas sobre o frame da camera")
    parser.add_argument("--camera", default="cam26", choices=CAMERA_NAMES, help="Qual camera usar")
    parser.add_argument("--hooks", default=None, help="JSON de calibracao (padrao: local_cv/hooks_<camera>_<n>.json)")
    parser.add_argument("--car-hooks", type=int, default=None,
                         help="Quantidade de ganchos do carro na linha (8 ou 11)")
    parser.add_argument("--api-url", default=os.environ.get("PARTS_API_URL", DEFAULT_API_URL))
    parser.add_argument("--out", default=None, help="Arquivo de saida (padrao: capturas/overlay_<camera>.jpg)")
    parser.add_argument("--pad-x", type=int, default=60, help="Folga horizontal da caixa da peca, em pixels")
    parser.add_argument("--part-height", type=int, default=320,
                         help="Altura da caixa abaixo dos ganchos (a peca fica pendurada embaixo)")
    parser.add_argument("--watch", action="store_true", help="Fica atualizando continuamente")
    parser.add_argument("--interval", type=float, default=5.0, help="Segundos entre atualizacoes com --watch")
    args = parser.parse_args()

    rtsp_url = camera_url(args.camera)

    out_path = args.out or f"capturas/overlay_{args.camera}.jpg"
    client = PartsClient(args.api_url)

    if not args.watch:
        render_once(client, rtsp_url, args.camera, args.hooks, args.car_hooks,
                     out_path, args.pad_x, args.part_height)
        return

    import time
    print(f"Atualizando a cada {args.interval}s. Ctrl+C para parar.")
    while True:
        try:
            render_once(client, rtsp_url, args.camera, args.hooks, args.car_hooks,
                         out_path, args.pad_x, args.part_height)
        except Exception as exc:
            print(f"Erro: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
