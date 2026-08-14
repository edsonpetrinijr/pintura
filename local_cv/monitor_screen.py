"""
Tela de monitoramento ao vivo dos ganchos.

Mostra o frame da camera com a deteccao local desenhada e, ao lado, um painel
comparando gancho a gancho a deteccao local com o que a API informa. Serve para
acompanhar a linha e, principalmente, para enxergar ONDE a deteccao erra.

A leitura passa por um estabilizador temporal (ver stability.py): decide pela
mediana dos ultimos frames e usa histerese, para que um gancho em cima do
limiar pare de piscar. Cada gancho mostra tambem a CERTEZA da leitura - quando
nao da para afirmar, a tela diz INCERTO em vez de fingir uma resposta.

Teclas:
    q / ESC   sair
    c         troca de camera (so as que tem calibracao)
    g         gravar snapshot em capturas/
    + / -     ajusta o limiar de ocupacao em 0.5
    [ / ]     ajusta a margem da histerese (zona morta) em 0.25
    a         liga/desliga a comparacao com a API
    j         mostra/esconde as janelas de analise
    p         pausa
    l         entra/sai do MODO ROTULO (pausa sozinho ao entrar). Dentro dele,
              clique numa linha do gancho no painel INVERTE a leitura daquele
              gancho (usa quando o detector ou o modelo errou)
    s         salva os recortes do frame atual em dataset_ocupacao/ (so
              dentro do modo rotulo) - vira dado de treino do modelo

Coluna MODELO no painel: previsao do classificador treinavel (ocupado/vazio),
se ja existir local_cv/modelo_ocupacao.joblib (treinar com treinar_ocupacao.py).
So informativo por enquanto - o veredito ainda usa a heuristica de bordas.
Fica em laranja quando discorda da heuristica: e o caso mais valioso pra rotular.

Uso:
    python local_cv/monitor_screen.py
    python local_cv/monitor_screen.py --camera cabine --api-interval 5
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import (SeletorDeCalibracao, available_calibrations,
                         load_thresholds)
from cameras import CAMERA_NAMES, camera_url, configured_cameras, label, vista_irma
from local_cv.detect_hooks_local import analyze
from local_cv.fusion import fundir
from local_cv.modelo_ocupacao import carregar_modelo, prever
from local_cv.stability import BAIXA, HookStabilizer
from local_cv.tabela_ganchos import carregar as carregar_tabela_ganchos
from local_cv.tabela_ganchos import checar_vao
from parts.parts_client import PartsClient

DEFAULT_API_URL = "http://b8wdwisep02.brazil.cat.com:3030/PartBldYJSON"

VERDE = (0, 200, 0)
VERMELHO = (0, 0, 255)
LARANJA = (0, 165, 255)
AMARELO = (0, 220, 220)
CINZA = (60, 60, 60)
CINZA_CLARO = (150, 150, 150)
BRANCO = (255, 255, 255)
CIANO = (255, 255, 0)

PAINEL_W = 620
FONTE = cv2.FONT_HERSHEY_SIMPLEX
PASTA_ROTULOS_PADRAO = "dataset_ocupacao"


def desenhar_frame(frame, results, vereditos, mostrar_janelas):
    for r in results:
        # Gancho vindo so da vista irma: nao tem coordenada NESTA imagem (ver
        # fusion.fundir). Ele aparece no painel, mas nao da para marcar aqui.
        if r.get("point") is None:
            continue

        marca, cor = vereditos[r["id"]]
        incerto = marca in ("?", "???")

        if mostrar_janelas:
            x1, y1, x2, y2 = r["window"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        cv2.circle(frame, r["point"], 9, (0, 0, 0), -1)
        if incerto:
            # Anel vazado: leitura sem confianca, da para ver de longe.
            cv2.circle(frame, r["point"], 7, cor, 2)
        else:
            cv2.circle(frame, r["point"], 6, cor, -1)

        rotulo = f"{r['id']}?" if incerto else str(r["id"])
        cv2.putText(frame, rotulo, (r["point"][0] + 10, r["point"][1] - 8),
                    FONTE, 0.6, cor, 2)
    return frame


def avaliar(results, api_hooks, vagas: int) -> dict:
    """Julga a leitura de cada gancho contra a API. Retorna {id: (marca, cor)}.

    A comparacao NAO e simetrica: "API diz ocupado" e informacao firme, mas
    "API nao lista" nao prova gancho vazio, porque peca sem programa de robo
    nao ganha posicao.

    O detalhe que faz diferenca e que a API DIZ quantas pecas estao nessa
    situacao - sao os registros que chegam com o campo hook vazio. Esse numero
    (`vagas`) limita quantas deteccoes a mais podem ser justificadas. Se nao ha
    nenhuma peca sem gancho, entao gancho detectado que a API nao lista e falso
    positivo nosso, e a tela precisa dizer ERRO em vez de dar de ombros.

    Quando ha vagas, o beneficio da duvida vai para as deteccoes de maior
    score: sao as mais provaveis de serem peca de verdade. As demais viram ERRO.

      OK    - concordam
      ERRO  - a API afirma ocupado e nao vimos, ou vimos algo que nao existe
      ?     - deteccao a mais que cabe numa peca sem programa de robo
      ???   - leitura sem confianca, nao da para julgar
    """
    if api_hooks is None:
        return {r["id"]: ("-", CINZA_CLARO) for r in results}

    vereditos = {}
    extras = []
    for r in results:
        if r["certeza"] == BAIXA:
            vereditos[r["id"]] = ("???", AMARELO)
            continue

        diz_api = r["id"] in api_hooks
        if diz_api == r["occupied"]:
            vereditos[r["id"]] = ("OK", VERDE)
        elif diz_api:
            vereditos[r["id"]] = ("ERRO", VERMELHO)
        else:
            extras.append(r)

    # Deteccoes a mais: so as `vagas` mais fortes ganham o beneficio da duvida.
    extras.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(extras):
        vereditos[r["id"]] = ("?", AMARELO) if i < vagas else ("ERRO", VERMELHO)
    return vereditos


def desenhar_painel(altura, results, vereditos, api_hooks, camera, calib, confianca,
                     encaixe, threshold, margem, pecas, vagas, stats, pausado,
                     aprendidos, confiavel, alertas_vao=(), modo_rotulo=False,
                     correcoes=None):
    correcoes = correcoes or {}
    painel = np.full((altura, PAINEL_W, 3), 25, np.uint8)
    y = 34
    linhas_y = {}  # hook_id -> (y0, y1) em coordenadas DO PAINEL, pro clique do mouse

    def linha(texto, cor=BRANCO, escala=0.5, passo=24):
        nonlocal y
        cv2.putText(painel, texto, (14, y), FONTE, escala, cor, 1, cv2.LINE_AA)
        y += passo

    linha(datetime.now().strftime("%d/%m/%Y  %H:%M:%S"), BRANCO, 0.62, 26)
    linha(f"camera: {camera}   ('c' troca)", CINZA_CLARO, 0.46, 26)
    if not confiavel:
        linha("CARRO EM MOVIMENTO", AMARELO, 0.6, 26)
        linha("nao confie na leitura", AMARELO, 0.45, 24)
    if modo_rotulo:
        linha("MODO ROTULO - clique numa linha p/ corrigir, 's' salva", CIANO, 0.46, 24)
    y += 6

    cv2.line(painel, (14, y - 14), (PAINEL_W - 14, y - 14), CINZA, 1)
    cv2.putText(painel, "GANCHO  LEITURA    CONF  VISTA  MODELO   API",
                (14, y + 4), FONTE, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    y += 16
    cv2.line(painel, (14, y - 2), (PAINEL_W - 14, y - 2), CINZA, 1)
    y += 28

    for r in results:
        marca, cor = vereditos[r["id"]]
        corrigido = r["id"] in correcoes
        ocupado_efetivo = correcoes[r["id"]] if corrigido else r["occupied"]
        if r["certeza"] == BAIXA and not corrigido:
            leitura = "???"
        else:
            leitura = "OCUPADO" if ocupado_efetivo else "vazio"

        pct = 100 * r["concordantes"] / max(r["amostras"], 1)
        fontes = "+".join(label(f) for f in r.get("fontes", [])) or label(camera)

        previsao = r.get("modelo")  # (ocupado: bool, confianca 0..1) ou None
        if previsao is None:
            modelo_txt, modelo_cor = "-", CINZA_CLARO
        else:
            m_ocupado, m_conf = previsao
            modelo_txt = f"{'OCUP' if m_ocupado else 'vazio'} {100 * m_conf:.0f}%"
            modelo_cor = LARANJA if m_ocupado != r["occupied"] else CINZA_CLARO

        linhas_y[r["id"]] = (y - 26, y + 4)
        cor_linha = CIANO if corrigido else BRANCO
        cv2.putText(painel, f"{r['id']:>4}", (14, y), FONTE, 0.62, cor_linha, 1, cv2.LINE_AA)
        cv2.putText(painel, leitura, (95, y), FONTE, 0.58, CIANO if corrigido else cor,
                    1, cv2.LINE_AA)
        cv2.putText(painel, f"{pct:3.0f}%", (250, y), FONTE, 0.52,
                    CINZA_CLARO if pct >= 80 else AMARELO, 1, cv2.LINE_AA)
        cv2.putText(painel, fontes, (325, y), FONTE, 0.5,
                    LARANJA if r.get("divergencia") else CINZA_CLARO, 1, cv2.LINE_AA)
        cv2.putText(painel, modelo_txt, (400, y), FONTE, 0.48, modelo_cor, 1, cv2.LINE_AA)
        cv2.putText(painel, marca, (540, y), FONTE, 0.62, cor,
                    2 if marca == "ERRO" else 1, cv2.LINE_AA)
        y += 30

    y += 8
    cv2.line(painel, (14, y - 16), (PAINEL_W - 14, y - 16), CINZA, 1)

    if api_hooks is not None:
        marcas = [m for m, _ in vereditos.values()]
        erros = marcas.count("ERRO")
        julgados = len(marcas) - marcas.count("???")
        if not julgados:
            # "ERRO: 0" em verde aqui seria mentira: nao houve leitura nenhuma.
            linha("SEM LEITURA CONFIAVEL", AMARELO, 0.56, 26)
        else:
            linha(f"ERRO: {erros}    duvida: {marcas.count('?')}"
                  f"{f'    sem ler: {len(marcas) - julgados}' if julgados < len(marcas) else ''}",
                  VERDE if erros == 0 else VERMELHO, 0.56, 26)
        if stats["api_ocupados"]:
            linha(f"achamos {stats['detectados']}/{stats['api_ocupados']} do que a "
                  f"API confirma", CINZA_CLARO, 0.44)

    y += 4
    linha(f"{os.path.basename(calib)}  encaixe {100 * encaixe:.0f}%"
          f"  (conf {confianca:.1f})", (130, 130, 130), 0.42, 18)
    linha(f"limiar {threshold:.1f}  margem {margem:.2f}"
          f"{f'  aprendido em {aprendidos}' if aprendidos else ''}",
          (130, 130, 130), 0.42, 18)
    if pausado:
        linha("PAUSADO", LARANJA, 0.55, 22)

    if pecas:
        y += 4
        for p in pecas[:3]:
            linha(f"  {p}", (140, 140, 140), 0.4, 17)

    if alertas_vao:
        y += 4
        linha("VAO NUNCA VISTO PRA ESSA PECA:", LARANJA, 0.42, 18)
        for a in alertas_vao[:3]:
            linha(f"  {a}", LARANJA, 0.4, 17)

    rodape = altura - 26
    if vagas:
        texto = (f"{vagas} peca(s) na API sem gancho atribuido.",
                 "Ate ai as deteccoes a mais viram '?'.")
    else:
        texto = ("Nenhuma peca sem gancho na API.",
                 "Entao deteccao a mais e erro nosso.")
    for i, t in enumerate(texto):
        cv2.putText(painel, t, (14, rodape + 14 * i), FONTE, 0.38,
                    (120, 120, 120), 1, cv2.LINE_AA)
    return painel, linhas_y


def salvar_rotulos(pasta, frame, camera, calib, results, correcoes):
    """Grava um recorte por gancho (leitura ja corrigida onde o usuario clicou)
    em dataset_ocupacao/{ocupado,vazio}/ + uma linha no CSV. So grava ganchos
    com certeza alta (leitura sem confianca nao vira dado de treino, a nao ser
    que o usuario tenha corrigido manualmente)."""
    os.makedirs(os.path.join(pasta, "ocupado"), exist_ok=True)
    os.makedirs(os.path.join(pasta, "vazio"), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    calib_nome = os.path.splitext(os.path.basename(calib))[0]
    csv_path = os.path.join(pasta, "rotulos.csv")
    novo = not os.path.exists(csv_path)
    gravados = 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["timestamp", "camera", "calibracao", "gancho", "ocupado",
                        "score", "corrigido_manual", "imagem"])
        for r in results:
            corrigido = r["id"] in correcoes
            if r["certeza"] == BAIXA and not corrigido:
                continue
            ocupado = correcoes[r["id"]] if corrigido else r["occupied"]
            x1, y1, x2, y2 = r["window"]
            x1, y1 = max(x1, 0), max(y1, 0)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            sub = "ocupado" if ocupado else "vazio"
            nome = f"{ts}_{camera}_{calib_nome}_g{r['id']:02d}.jpg"
            cv2.imwrite(os.path.join(pasta, sub, nome), crop)
            w.writerow([ts, camera, calib_nome, r["id"], int(ocupado),
                        f"{r['score']:.2f}", int(corrigido), f"{sub}/{nome}"])
            gravados += 1
    return gravados


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Tela de monitoramento dos ganchos")
    parser.add_argument("--camera", default="cabine", choices=CAMERA_NAMES)
    parser.add_argument("--car-hooks", type=int, default=None)
    parser.add_argument("--hooks", default=None)
    parser.add_argument("--api-url", default=os.environ.get("PARTS_API_URL", DEFAULT_API_URL))
    parser.add_argument("--api-interval", type=float, default=10.0,
                         help="Segundos entre consultas a API")
    parser.add_argument("--threshold", type=float, default=4.5)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=50)
    parser.add_argument("--max-size", type=int, default=50)
    parser.add_argument("--drop", type=float, default=0.8)
    parser.add_argument("--janela", type=int, default=45,
                         help="Quantos frames entram na mediana que estabiliza a leitura. "
                              "A ~25 fps, 45 frames cobrem uns 2s, o bastante para um "
                              "operador atravessar a frente do gancho sem mudar a leitura")
    parser.add_argument("--margem", type=float, default=1.0,
                         help="Zona morta em volta do limiar (histerese)")
    parser.add_argument("--encaixe-min", type=float, default=0.7,
                         help="Fracao do pico recente da propria calibracao abaixo da "
                              "qual a cena esta em transicao e a leitura toda e "
                              "marcada como nao confiavel. Limiar absoluto nao serve "
                              "aqui: a medida muda com carro, luz e fumaca")
    parser.add_argument("--largura", type=int, default=1280, help="Largura da imagem na tela")
    parser.add_argument("--fusao", action="store_true",
                         help="Cruza a leitura com a segunda vista da cabine (.46). "
                              "DESLIGADO por padrao: o peso de cada vista sai so da "
                              "geometria dos ganchos, nunca de quanto ela esta "
                              "enxergando agora, entao uma vista cega por neblina de "
                              "tinta continua votando com forca total e anula a boa")
    parser.add_argument("--dataset-rotulos", default=PASTA_ROTULOS_PADRAO,
                         help="Pasta onde 's' salva os recortes rotulados (modo 'l')")
    args = parser.parse_args()

    # Trocar de camera so faz sentido para quem tem calibracao: sem ela nao ha
    # o que ler, e pick_calibration abortaria a tela inteira.
    configuradas = configured_cameras()
    rotativas = [c for c in CAMERA_NAMES if c in configuradas and available_calibrations(c)]
    if args.camera not in rotativas:
        rotativas.append(args.camera)

    camera = args.camera
    url = camera_url(camera)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir a camera: {url}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    client = PartsClient(args.api_url)
    janela = "Ganchos"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    modelo_ocupacao = carregar_modelo()
    print(f"modelo de ocupacao: {'carregado' if modelo_ocupacao else 'ainda nao treinado'}"
          f"{'' if modelo_ocupacao else ' (rode local_cv/treinar_ocupacao.py depois de rotular)'}")

    # Estado do modo de rotulagem, mutavel pelo callback de mouse. linhas_y e
    # largura_imagem sao atualizados a cada frame (a janela do painel se move
    # conforme o painel e a imagem sao redesenhados).
    estado_rotulo = {"ativo": False, "correcoes": {}, "linhas_y": {},
                     "leituras": {}, "largura_imagem": 0}

    def ao_clicar(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or not estado_rotulo["ativo"]:
            return
        x_painel = x - estado_rotulo["largura_imagem"]
        if x_painel < 0:
            return
        for hook_id, (y0, y1) in estado_rotulo["linhas_y"].items():
            if y0 <= y <= y1:
                if hook_id in estado_rotulo["correcoes"]:
                    del estado_rotulo["correcoes"][hook_id]
                else:
                    estado_rotulo["correcoes"][hook_id] = not estado_rotulo["leituras"].get(hook_id, False)
                break

    cv2.setMouseCallback(janela, ao_clicar)

    # Segunda vista da mesma fila. Vale abrir mesmo com imagem pior: o que ela
    # acrescenta e angulo, nao nitidez - ganchos que a primeira ve espremidos
    # no fundo ela ve de lado.
    irma, cap2 = None, None
    candidata = vista_irma(camera) if args.fusao else None
    if candidata and candidata in configuradas and available_calibrations(candidata):
        cap2 = cv2.VideoCapture(camera_url(candidata))
        if cap2.isOpened():
            cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            irma = candidata
            print(f"fundindo com '{irma}'")
        else:
            # Costuma ser limite de streams simultaneos da camera (RTSP 453).
            cap2.release()
            cap2 = None
            print(f"nao consegui abrir '{candidata}'; seguindo so com '{camera}'")

    threshold = args.threshold
    margem = args.margem
    usar_api = True
    mostrar_janelas = True
    pausado = False

    api_hooks, pecas, vagas, ultima_api = set(), [], 0, 0.0
    alertas_vao = []
    # So mede contra o que a API afirma estar ocupado: e o unico rotulo
    # confiavel dela. "API nao lista" pode ser peca sem programa de robo.
    stats = {"api_ocupados": 0, "detectados": 0}
    tabela_ganchos = carregar_tabela_ganchos()
    frame = None

    estabilizador = HookStabilizer(args.janela, margem)
    estabilizador2 = HookStabilizer(args.janela, margem)
    seletor = SeletorDeCalibracao(camera, args.hooks, args.car_hooks,
                                  fracao=args.encaixe_min)
    seletor2 = SeletorDeCalibracao(irma, fracao=args.encaixe_min) if irma else None
    calib_atual = calib2_atual = None
    limiares = {}
    vista2 = None

    print(f"Tela aberta em '{camera}'. 'c' troca de camera ({rotativas}), 'q' sai.")

    while True:
        if not pausado:
            ok, novo = cap.read()
            if not ok:
                print("Frame perdido, reconectando...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(url)
                continue
            frame = novo

        if frame is None:
            continue

        if usar_api and time.time() - ultima_api > args.api_interval:
            # So a chamada de rede entra no try: erro de formatacao aqui e bug,
            # e nao deve ser mascarado como "API indisponivel".
            try:
                records = client.fetch()
            except requests.RequestException as exc:
                print(f"API indisponivel: {exc}")
                records = None

            if records is not None:
                api_hooks = {h for r in records for h in r.hooks}
                # Peca que a API lista sem gancho: existe no carro, mas sem
                # posicao. E o unico numero que justifica deteccao a mais.
                vagas = sum(1 for r in records if not r.has_hooks)
                pecas = [f"{r.part_number} ({r.figure or '?'}) g:"
                          f"{','.join(str(h) for h in r.hooks) or 'SEM GANCHO'}"
                          for r in records]
                # Checagem barata: o vao que a PROPRIA API relata pra esse
                # part_number ja apareceu antes no historico da planilha de
                # programas? Fora do historico = alerta, nao prova de peca
                # errada sozinho (ver local_cv/tabela_ganchos.py).
                alertas_vao = []
                for r in records:
                    if not r.hooks:
                        continue
                    chk = checar_vao(r.part_number, r.hooks, tabela_ganchos)
                    if chk["esperado"] is False:
                        alertas_vao.append(
                            f"{r.part_number} ({r.figure or '?'}) gancho "
                            f"{chk['faixa_detectada']} - conhecidas: "
                            f"{sorted(chk['faixas_conhecidas'])}")
            ultima_api = time.time()

        trabalho = frame.copy()
        calib, hooks, confianca, encaixe = seletor.escolher(trabalho)

        if calib != calib_atual:
            # Carro diferente: a geometria mudou, entao o historico anterior
            # nao vale mais para os mesmos ids de gancho.
            estabilizador = HookStabilizer(args.janela, margem)
            limiares = load_thresholds(calib)
            calib_atual = calib
            print(f"calibracao: {os.path.basename(calib)}"
                  f"{f' (limiares aprendidos: {sorted(limiares)})' if limiares else ''}")

        brutos = analyze(trabalho, hooks, None, threshold, args.factor,
                          args.min_size, args.max_size, args.drop, None, None, None)
        estabilizador.margem = margem
        results = estabilizador.aplicar(brutos, threshold, limiares)

        # Confianca baixa significa que os pontos calibrados nao estao caindo
        # sobre os ganchos: carro entrando/saindo, porta, fumaca. Nessa hora a
        # leitura inteira nao vale, e nao so um gancho ou outro.
        confiavel = seletor.confiavel(encaixe)

        vista2 = None
        if cap2 is not None and not pausado:
            ok2, frame2 = cap2.read()
            if ok2:
                calib2, hooks2, conf2, encaixe2 = seletor2.escolher(frame2)
                if calib2 != calib2_atual:
                    estabilizador2 = HookStabilizer(args.janela, margem)
                    calib2_atual = calib2
                estabilizador2.margem = margem
                limiares2 = load_thresholds(calib2)
                vista2 = {
                    "results": estabilizador2.aplicar(
                        analyze(frame2, hooks2, None, threshold, args.factor,
                                args.min_size, args.max_size, args.drop, None, None, None),
                        threshold, limiares2),
                    "hooks": hooks2, "limiares": limiares2,
                    "threshold": threshold, "confianca": encaixe2,
                }

        if vista2 is not None:
            results = fundir({
                camera: {"results": results, "hooks": hooks, "limiares": limiares,
                          "threshold": threshold, "confianca": encaixe},
                irma: vista2,
            }, args.encaixe_min, principal=camera)
            # Com duas vistas, "cena confiavel" e ter pelo menos uma valendo.
            confiavel = confiavel or vista2["confianca"] >= args.encaixe_min

        if not confiavel:
            results = [{**r, "certeza": BAIXA} for r in results]

        if modelo_ocupacao is not None:
            for r in results:
                x1, y1, x2, y2 = r["window"]
                crop = trabalho[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                r["modelo"] = prever(modelo_ocupacao, crop)

        comparar = api_hooks if usar_api else None
        if comparar is not None and not pausado:
            for r in results:
                if r["certeza"] == BAIXA or r["id"] not in comparar:
                    continue
                stats["api_ocupados"] += 1
                stats["detectados"] += r["occupied"]

        vereditos = avaliar(results, comparar, vagas)
        desenhar_frame(trabalho, results, vereditos, mostrar_janelas)

        escala = args.largura / trabalho.shape[1]
        imagem = cv2.resize(trabalho, (args.largura, int(trabalho.shape[0] * escala)))
        painel, linhas_y = desenhar_painel(
            imagem.shape[0], results, vereditos, comparar,
            camera, os.path.basename(calib), confianca,
            encaixe, threshold, margem, pecas,
            vagas, stats, pausado, len(limiares), confiavel,
            alertas_vao, estado_rotulo["ativo"], estado_rotulo["correcoes"])
        estado_rotulo["linhas_y"] = linhas_y
        estado_rotulo["leituras"] = {r["id"]: r["occupied"] for r in results}
        estado_rotulo["largura_imagem"] = imagem.shape[1]
        cv2.imshow(janela, np.hstack([imagem, painel]))

        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), 27):
            break
        elif tecla == ord("l"):
            estado_rotulo["ativo"] = not estado_rotulo["ativo"]
            if estado_rotulo["ativo"]:
                pausado = True
            else:
                estado_rotulo["correcoes"] = {}
        elif tecla == ord("s") and estado_rotulo["ativo"]:
            n = salvar_rotulos(args.dataset_rotulos, trabalho, camera,
                                os.path.basename(calib), results, estado_rotulo["correcoes"])
            print(f"rotulos salvos: {n} recorte(s) em {args.dataset_rotulos}/")
            estado_rotulo["correcoes"] = {}
        elif tecla == ord("c") and len(rotativas) > 1:
            camera = rotativas[(rotativas.index(camera) + 1) % len(rotativas)]

            # Solta as DUAS antes de abrir qualquer coisa: a camera que vai
            # virar principal e justamente a que estava aberta como vista
            # irma, e ela recusa dois streams (RTSP 453).
            cap.release()
            if cap2 is not None:
                cap2.release()
            cap2, irma, calib2_atual, seletor2 = None, None, None, None

            url = camera_url(camera)
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            nova_irma = vista_irma(camera) if args.fusao else None
            if nova_irma and nova_irma in configuradas and available_calibrations(nova_irma):
                cap2 = cv2.VideoCapture(camera_url(nova_irma))
                if cap2.isOpened():
                    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    irma = nova_irma
                    estabilizador2 = HookStabilizer(args.janela, margem)
                    seletor2 = SeletorDeCalibracao(irma, fracao=args.encaixe_min)
                else:
                    cap2.release()
                    cap2 = None

            # Outra camera, outra geometria: historico e taxa acumulada da
            # anterior nao valem para esta.
            estabilizador = HookStabilizer(args.janela, margem)
            # --hooks aponta um ARQUIVO, e candidate_paths o devolve sem sequer
            # olhar a camera. Levar esse forcado na troca desenha os pontos da
            # camera de origem sobre a imagem da nova. --car-hooks e pior: se a
            # nova camera nao tiver aquela contagem, candidate_paths encerra o
            # programa no meio do laco. Ambos so valem para a camera em que
            # foram pedidos; nas outras, volta a escolha automatica.
            forcado = args.hooks if camera == args.camera else None
            forcado_n = args.car_hooks if camera == args.camera else None
            seletor = SeletorDeCalibracao(camera, forcado, forcado_n,
                                          fracao=args.encaixe_min)
            calib_atual = None
            stats = {"api_ocupados": 0, "detectados": 0}
            frame = None
            # Sem despausar, o laco ficaria preso esperando um frame que nao
            # viria, e a tela travaria sem aceitar tecla.
            pausado = False
            print(f"camera: {camera}" + (f" (fundindo com {irma})" if irma else ""))
        elif tecla == ord("g"):
            destino = f"capturas/monitor_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            os.makedirs("capturas", exist_ok=True)
            cv2.imwrite(destino, np.hstack([imagem, painel]))
            print(f"gravado: {destino}")
        elif tecla in (ord("+"), ord("=")):
            threshold += 0.5
        elif tecla in (ord("-"), ord("_")):
            threshold = max(0.0, threshold - 0.5)
        elif tecla == ord("]"):
            margem += 0.25
        elif tecla == ord("["):
            margem = max(0.0, margem - 0.25)
        elif tecla == ord("a"):
            usar_api = not usar_api
        elif tecla == ord("j"):
            mostrar_janelas = not mostrar_janelas
        elif tecla == ord("p"):
            pausado = not pausado

    cap.release()
    if cap2 is not None:
        cap2.release()
    cv2.destroyAllWindows()
    if stats["api_ocupados"]:
        print(f"detectamos {stats['detectados']} de {stats['api_ocupados']} ganchos "
              f"que a API confirmou ocupados "
              f"({stats['detectados'] / stats['api_ocupados'] * 100:.0f}%)")


if __name__ == "__main__":
    main()
