"""
Detecta localmente (sem Maximo) se cada gancho calibrado esta OCUPADO ou VAZIO,
analisando a regiao logo ABAIXO de cada ponto de gancho (onde a peca fica
pendurada). Opcionalmente compara o resultado com o que a API de pecas informa.

A janela analisada escala com a perspectiva: ganchos mais proximos da camera
ficam mais espacados entre si, entao recebem uma janela maior.

Modos de classificacao:
  - densidade de bordas (padrao): peca pendurada gera muitas bordas; fundo vazio
    (teto/parede) gera poucas.
  - diferenca contra um frame de referencia com a linha vazia (--background),
    mais confiavel quando disponivel.

Uso:
    python local_cv/detect_hooks_local.py --camera cam26
    python local_cv/detect_hooks_local.py --camera cam26 --compare-api
    python local_cv/detect_hooks_local.py --image capturas/ref_cam26.jpg --hooks local_cv/hooks_cam26.json
"""
import argparse
import os
import sys

import cv2
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import load_hooks, pick_calibration
from cameras import CAMERA_NAMES, camera_url
from parts.parts_client import PartsClient

COLOR_OCCUPIED = (0, 0, 255)
COLOR_EMPTY = (0, 200, 0)
COLOR_DISAGREE = (0, 165, 255)


def patch_size_for(hook: dict, hooks: list[dict], factor: float,
                    min_size: int, max_size: int) -> int:
    """Estima o tamanho da janela pela distancia ao gancho vizinho mais proximo."""
    others = [h for h in hooks if h["id"] != hook["id"]]
    if not others:
        return min_size

    spacing = min(abs(h["x"] - hook["x"]) for h in others)
    return int(np.clip(spacing * factor, min_size, max_size))


def hook_window(hook: dict, size: int, drop: float, drop_px: int | None,
                 frame_shape, win_w: int | None = None,
                 win_h: int | None = None) -> tuple[int, int, int, int]:
    """Janela abaixo do gancho, onde a peca fica pendurada.

    Por padrao e quadrada com lado `size`. Com --window-w/--window-h vira um
    retangulo, util para pegar uma faixa vertical estreita (corrente descendo).
    """
    w = win_w if win_w is not None else size
    h = win_h if win_h is not None else size

    cx = hook["x"]
    offset = drop_px if drop_px is not None else int(size * drop)
    cy = hook["y"] + offset

    x1 = max(0, cx - w // 2)
    x2 = min(frame_shape[1], cx + w // 2)
    y1 = max(0, cy - h // 2)
    y2 = min(frame_shape[0], cy + h // 2)
    return x1, y1, x2, y2


def score_window(frame, window, background) -> tuple[float, str]:
    x1, y1, x2, y2 = window
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0, "vazia"

    if background is not None:
        bg_roi = background[y1:y2, x1:x2]
        diff = cv2.absdiff(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY))
        return float(np.mean(diff)), "diff"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    density = float(np.count_nonzero(edges)) / edges.size * 100.0
    return density, "bordas"


def analyze(frame, hooks: list[dict], background, threshold: float,
             factor: float, min_size: int, max_size: int, drop: float,
             drop_px: int | None, win_w: int | None = None,
             win_h: int | None = None) -> list[dict]:
    results = []
    for hook in hooks:
        size = patch_size_for(hook, hooks, factor, min_size, max_size)
        window = hook_window(hook, size, drop, drop_px, frame.shape, win_w, win_h)
        score, mode = score_window(frame, window, background)
        results.append({
            "id": hook["id"],
            "point": (hook["x"], hook["y"]),
            "window": window,
            "score": score,
            "mode": mode,
            "occupied": score > threshold,
        })
    return results


def draw_windows_only(frame, results: list[dict]):
    """Modo de ajuste: mostra so onde cada janela caiu, sem classificar."""
    for r in results:
        x1, y1, x2, y2 = r["window"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(frame, r["point"], 8, (0, 0, 0), -1)
        cv2.circle(frame, r["point"], 6, (0, 255, 255), -1)
        cv2.line(frame, r["point"], ((x1 + x2) // 2, (y1 + y2) // 2), (255, 255, 0), 1)
        cv2.putText(frame, str(r["id"]), (x1 + 4, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, "MODO AJUSTE - posicione as janelas com --drop-px / --factor",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def draw(frame, results: list[dict], api_hooks: set[int] | None):
    for r in results:
        x1, y1, x2, y2 = r["window"]
        occupied = r["occupied"]

        color = COLOR_OCCUPIED if occupied else COLOR_EMPTY
        label = "OCUPADO" if occupied else "VAZIO"

        if api_hooks is not None:
            api_says = r["id"] in api_hooks
            if api_says != occupied:
                color = COLOR_DISAGREE
                label += f" (API: {'OCUPADO' if api_says else 'VAZIO'})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, r["point"], 8, (0, 0, 0), -1)
        cv2.circle(frame, r["point"], 6, color, -1)

        text = f"{r['id']}: {label} {r['score']:.1f}"
        cv2.putText(frame, text, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    ocupados = sum(1 for r in results if r["occupied"])
    resumo = f"ganchos: {len(results)}  ocupados: {ocupados}  vazios: {len(results) - ocupados}"
    if api_hooks is not None:
        divergentes = sum(1 for r in results if (r["id"] in api_hooks) != r["occupied"])
        resumo += f"  |  divergencias vs API: {divergentes}"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, resumo, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def capture_frame(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Nao consegui capturar frame de: {rtsp_url}")
    return frame


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Detecta ocupado/vazio em cada gancho calibrado")
    parser.add_argument("--camera", default="cam26", choices=CAMERA_NAMES)
    parser.add_argument("--image", help="Analisa uma imagem salva em vez de capturar da camera")
    parser.add_argument("--hooks", help="JSON de calibracao (padrao: local_cv/hooks_<camera>_<n>.json)")
    parser.add_argument("--car-hooks", type=int, default=None,
                         help="Quantidade de ganchos do carro na linha (8 ou 11)")
    parser.add_argument("--background", help="Frame de referencia com a linha vazia")
    parser.add_argument("--threshold", type=float, default=None,
                         help="Limiar de ocupacao (padrao: 4.5 para bordas, 15.0 para --background)")
    parser.add_argument("--factor", type=float, default=0.9,
                         help="Tamanho da janela como fracao do espacamento entre ganchos")
    parser.add_argument("--min-size", type=int, default=40, help="Tamanho minimo da janela em pixels")
    parser.add_argument("--max-size", type=int, default=260, help="Tamanho maximo da janela em pixels")
    parser.add_argument("--drop", type=float, default=0.9,
                         help="Quanto a janela desce abaixo do gancho, em multiplos do tamanho")
    parser.add_argument("--drop-px", type=int, default=None,
                         help="Deslocamento vertical fixo em pixels (sobrepoe --drop)")
    parser.add_argument("--window-w", type=int, default=None,
                         help="Largura fixa da janela (sobrepoe o tamanho automatico)")
    parser.add_argument("--window-h", type=int, default=None,
                         help="Altura fixa da janela; use alta e estreita para pegar a corrente")
    parser.add_argument("--tune", action="store_true",
                         help="Modo de ajuste: so desenha onde as janelas caem, sem classificar")
    parser.add_argument("--compare-api", action="store_true", help="Compara com a API de pecas")
    parser.add_argument("--out", help="Arquivo de saida (padrao: capturas/hooks_local_<camera>.jpg)")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Nao consegui abrir: {args.image}")
    else:
        frame = capture_frame(camera_url(args.camera))

    _, hooks, _ = pick_calibration(frame, args.camera, args.hooks, args.car_hooks)

    background = None
    if args.background:
        background = cv2.imread(args.background)
        if background is None:
            raise SystemExit(f"Nao consegui abrir o background: {args.background}")

    threshold = args.threshold
    if threshold is None:
        # 4.5 vem da separacao medida na cabine: vazios ficaram <= 3.1 e ocupados >= 5.6
        threshold = 15.0 if background is not None else 4.5

    results = analyze(frame, hooks, background, threshold,
                       args.factor, args.min_size, args.max_size, args.drop, args.drop_px,
                       args.window_w, args.window_h)

    if args.tune:
        out_path = args.out or f"capturas/hooks_tune_{args.camera}.jpg"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, draw_windows_only(frame, results))
        print(f"Modo ajuste. Janelas desenhadas em: {out_path}")
        return

    api_hooks = None
    if args.compare_api:
        client = PartsClient(os.environ.get("PARTS_API_URL",
                                             "http://b8wdwisep02.brazil.cat.com:3030/PartBldYJSON"))
        api_hooks = {h for record in client.fetch() for h in record.hooks}
        print(f"API diz ocupados: {sorted(api_hooks) or 'nenhum'}")

    for r in results:
        status = "OCUPADO" if r["occupied"] else "VAZIO"
        linha = f"gancho {r['id']:>2}: {status:<8} score={r['score']:6.1f} ({r['mode']})"
        if api_hooks is not None:
            api_says = r["id"] in api_hooks
            if api_says != r["occupied"]:
                linha += f"  <-- DIVERGE (API diz {'OCUPADO' if api_says else 'VAZIO'})"
        print(linha)

    out_path = args.out or f"capturas/hooks_local_{args.camera}.jpg"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, draw(frame, results, api_hooks))
    print(f"\nImagem anotada salva em: {out_path}")


if __name__ == "__main__":
    main()
