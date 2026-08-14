"""Sobrepoe a silhueta do CAD na imagem ao vivo da camera, ajustavel na mao.

Serve para duas coisas ao mesmo tempo. A primeira e conferir com o olho se o
modelo 3D consegue explicar o que a camera ve: numero de IoU nao diz ONDE errou,
e a sobreposicao diz. A segunda e achar na mao a pose certa quando o casamento
automatico erra, e assim descobrir se o problema e o banco de angulos (pose que
nao existe nele) ou a segmentacao (a mascara da camera esta errada).

O 3D e desenhado como contorno grosso mais preenchimento translucido: contorno
sozinho some no meio da fumaca da cabine, e preenchimento solido esconde a peca
que se quer comparar.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cameras import CAMERA_NAMES, camera_url, label
from calibration import load_hooks, pick_calibration
from local_cv import modelo3d
from local_cv.silhueta import silhuetas

EXTENSOES = ("*.step", "*.stp", "*.glb", "*.stl", "*.obj")
COR_3D = (0, 220, 255)      # amarelo: contrasta com a peca escura e com a cabine
COR_CAM = (0, 255, 120)     # verde: o que a segmentacao achou
RENDER = 320                # lado do quadro de render antes de reescalar

# Giro de 90 graus em torno do eixo VERTICAL da peca pendurada. A malha vem do
# pipeline JT/glTF (Teamcenter), que e Y-up, entao o eixo da corrente e o Y do
# modelo - nao o Z. A tecla 't' PRE-rotaciona a malha por isto (antes da pose),
# que e o que troca a face vista pela camera; girar o yaw (Z) com a camera no
# alto so rodava a silhueta no plano, sem mudar de lado. Ry(90) = colunas de x
# viram -z e z vira x.
VIRA_LADO = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0]])

# Pose de partida medida no 5756942 (LiftArm) na cabine, IoU 0.658. A camera e
# fixa no alto e a peca pende de corrente, entao a direcao de visada e a
# inclinacao no plano da imagem mal mudam de peca para peca. O que o operador
# escolhe e para que lado a peca aponta, e isso sai em passos de 90 graus.
# Por isso yaw comeca aqui e e o unico eixo que a busca por quadrante varre.
POSE_PADRAO = {"yaw": 118.0, "pitch": -56.0, "giro": 252.0}

# Focal DESLIGADA por padrao (0 = desligado). Com ela ligada, a geometria passa
# a depender de onde a peca esta no quadro e de quanto zoom ela tem, e as duas
# coisas invalidam o cache de render: arrastar 900 px virava 13 renders, ~2s de
# janela travada. A parte de fisica esta certa e medida - ver adiante -, mas
# custa interatividade, e ajustar pose na mao exige janela que responde.
#
# Para religar: --focal 1012
#   1012 px e a focal desta cabine, medida por local_cv/fuga.py sobre 2250
#   retas acumuladas de 9 frames (FOV horizontal 87 graus, repetivel em +-8 px).
#   Com ela, o render acompanha o desvio do eixo optico: medido com pose
#   verdadeira conhecida, uma peca a 40 graus do eixo e recuperada com 5 graus
#   de erro de yaw em vez de 26, e o IoU sobe de 0.506 para 0.824 a 28 graus.
#   Falta acertar POSE_PADRAO junto: ela foi medida com a focal desligada e tem
#   o desvio da posicao daquela peca embutido.
FOCAL = 0.0


def distancia_para(altura_px: float, focal: float = FOCAL) -> float:
    """Distancia camera-peca em raios da peca, que e o que modelo3d espera.

    Uma peca de meio-tamanho `raio` a distancia `d` aparece com `f*raio/d` px de
    meia-altura. Invertendo, `d/raio = 2*f/altura_px`. Ou seja: nao ha nada para
    escolher aqui - se a peca cresce na tela, ela esta mais perto, e a
    perspectiva tem que ficar mais forte na mesma medida.

    Sem focal devolve a constante do modulo: a perspectiva continua existindo
    (o render nao e ortografico), so nao acompanha o zoom.

    Arredondado em passos de 0.5 raio de proposito: o cache de vistas e por
    angulo E por distancia, e sem quantizar cada pixel de zoom invalidaria o
    cache inteiro.
    """
    if not focal:
        return modelo3d.DISTANCIA
    return max(1.5, round(2.0 * focal / max(altura_px, 1.0) * 2) / 2)


PASSO_DESVIO = 4.0   # graus


def desvio_para(cx: float, cy: float, forma, focal: float = FOCAL,
               passo: float = PASSO_DESVIO) -> tuple[float, float]:
    """Angulos horizontal e vertical entre a peca e o eixo optico.

    Com FOV de 87 graus, uma peca encostada na borda esta a atan(960/1012) = 43
    graus do centro. Ignorar isso e dizer que a camera ve a peca da mesma face
    esteja ela onde estiver, e o refinador acaba compensando no yaw: a pose
    gravada fica errada em ate dezenas de graus dependendo so de onde a peca
    estava no quadro.

    Sem focal devolve zero, ou seja o comportamento antigo: a peca e sempre
    renderizada como se estivesse no centro do quadro.

    Quantizado em passos de 4 graus porque o cache de render tem o desvio na
    chave: sem quantizar, arrastar a peca um pixel ja invalidaria a vista.
    """
    if not focal:
        return (0.0, 0.0)
    alt, larg = forma[:2]
    ax = np.degrees(np.arctan2(cx - larg / 2.0, focal))
    ay = np.degrees(np.arctan2(cy - alt / 2.0, focal))
    return (round(ax / passo) * passo, round(ay / passo) * passo)


def modelos_em(pasta: str) -> list[str]:
    achados = []
    for padrao in EXTENSOES:
        achados += glob.glob(os.path.join(pasta, padrao))
    return sorted(achados)


def rotulo(caminho: str) -> str:
    return os.path.splitext(os.path.basename(caminho))[0].split("-")[0].strip()


def renderizar(malha, yaw: float, pitch: float, giro: float,
               tamanho: int = RENDER, distancia: float = modelo3d.DISTANCIA,
               desvio=None) -> np.ndarray | None:
    """Silhueta recortada na propria caixa, ja com o giro no plano da imagem.

    O giro nao entra na rotacao 3D porque nao e grau de liberdade da peca: e a
    camera que esta torta e a peca que balanca pendurada. Girar a imagem depois
    e mais barato e nao polui o banco de angulos.
    """
    mask = modelo3d.silhueta(malha, yaw, pitch, tamanho, distancia, desvio)
    if giro:
        centro = (tamanho / 2.0, tamanho / 2.0)
        M = cv2.getRotationMatrix2D(centro, giro, 1.0)
        mask = cv2.warpAffine(mask, M, (tamanho, tamanho), flags=cv2.INTER_NEAREST)

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def renderizar_visivel(malha, yaw: float, pitch: float, giro: float,
                       tamanho: int = RENDER,
                       distancia: float = modelo3d.DISTANCIA, desvio=None):
    """Igual a renderizar(), mas devolve tambem o 3D sombreado para exibir.

    Silhueta chapada e dificil de julgar a olho: sem sombra nao da para dizer se
    a peca esta de frente ou de costas, nem onde estao os vaos. O casamento
    continua usando so a mascara; a cor e para o operador.
    """
    cor, mask = modelo3d.sombreado(malha, yaw, pitch, tamanho,
                                   distancia=distancia, desvio=desvio)
    if giro:
        M = cv2.getRotationMatrix2D((tamanho / 2.0, tamanho / 2.0), giro, 1.0)
        cor = cv2.warpAffine(cor, M, (tamanho, tamanho), flags=cv2.INTER_NEAREST)
        mask = cv2.warpAffine(mask, M, (tamanho, tamanho), flags=cv2.INTER_NEAREST)

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return cor[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def colocar(recorte: np.ndarray, forma, cx: int, cy: int, altura: int,
            espelho: bool) -> np.ndarray:
    """Poe o recorte do CAD no tamanho e na posicao pedidos, dentro do frame.

    Aceita mascara (2D) ou imagem colorida (3D), para que a mascara usada no
    casamento e o render mostrado na tela sofram exatamente a mesma
    transformacao - se divergirem, o numero na tela deixa de descrever o que o
    olho ve.
    """
    if espelho:
        recorte = cv2.flip(recorte, 1)

    alt, larg = recorte.shape[:2]
    escala = max(altura, 8) / alt
    nl, na = max(1, int(larg * escala)), max(1, int(alt * escala))
    novo = cv2.resize(recorte, (nl, na), interpolation=cv2.INTER_NEAREST)

    destino = forma[:2] if recorte.ndim == 2 else (forma[0], forma[1], 3)
    quadro = np.zeros(destino, np.uint8)
    x0, y0 = int(cx - nl / 2), int(cy - na / 2)
    # recorta o que passar da borda, senao arrastar a peca para fora quebra
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(quadro.shape[1], x0 + nl - sx0)
    y1 = min(quadro.shape[0], y0 + na - sy0)
    if x1 > x0 and y1 > y0:
        quadro[y0:y1, x0:x1] = novo[sy0:sy0 + y1 - y0, sx0:sx0 + x1 - x0]
    return quadro


def mascara_cheia(peca: dict, forma) -> np.ndarray:
    """Leva a mascara recortada da peca de volta para o tamanho do frame."""
    x, y, w, h = peca["caixa"]
    quadro = np.zeros(forma[:2], np.uint8)
    quadro[y:y + h, x:x + w] = peca["mask"]
    return quadro


def desenhar(frame, mask, cor, alpha: float, grossura: int = 2):
    """Preenchimento translucido mais contorno solido.

    Mistura so dentro da caixa da mascara: um addWeighted sobre os 1920x1080
    inteiros por peca, a cada frame, e o suficiente para a janela engasgar.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    janela, sub = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1] > 0
    janela[sub] = (alpha * np.array(cor) + (1 - alpha) * janela[sub]).astype(np.uint8)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contornos, -1, cor, grossura)


def desenhar_solido(frame, render, mask, alpha: float, grossura: int = 2):
    """Mistura o 3D sombreado na imagem, so onde ha peca."""
    if not np.count_nonzero(mask):
        return
    onde = mask > 0
    frame[onde] = (alpha * render[onde] + (1 - alpha) * frame[onde]).astype(np.uint8)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contornos, -1, COR_3D, grossura)


def painel(frame, linhas: list[str]):
    fundo = frame[:len(linhas) * 26 + 12, :520].copy()
    cv2.rectangle(fundo, (0, 0), (520, len(linhas) * 26 + 12), (0, 0, 0), -1)
    cv2.addWeighted(fundo, 0.55, frame[:fundo.shape[0], :520], 0.45,
                    0, frame[:fundo.shape[0], :520])
    for i, texto in enumerate(linhas):
        cv2.putText(frame, texto, (10, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)


PASSOS = {"yaw": 20.0, "pitch": 15.0, "giro": 15.0,
          "cx": 40.0, "cy": 40.0, "altura": 40.0}
LIMITES = {"pitch": (-89.0, 89.0), "altura": (20.0, 4000.0)}


def mapa_de_bordas(frame) -> np.ndarray:
    """Distancia de cada pixel ate a borda mais proxima da imagem.

    Existe porque cor nao separa a peca do fundo nesta cabine. Medido no mesmo
    dia e na mesma camera: a peca deu saturacao 227 num frame e 148 em outro, e
    148 e exatamente a saturacao da grade do piso (152) e da capa de robo (148).
    Borda nao depende de tinta: o contorno da peca contra o fundo aparece
    independente de ela estar crua, recem-pintada ou empoeirada.
    """
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # bilateral antes do Canny: a grade do piso e a fumaca geram borda fina em
    # todo canto, e sem suavizar preservando aresta o mapa vira ruido uniforme
    bordas = cv2.Canny(cv2.bilateralFilter(cinza, 7, 60, 60), 40, 120)
    return cv2.distanceTransform(255 - bordas, cv2.DIST_L2, 3)


def nota_de_bordas(mask: np.ndarray, mapa: np.ndarray) -> float:
    """Quao bem o contorno do CAD assenta sobre as bordas da imagem, de 0 a 1."""
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return 0.0
    pts = np.vstack(contornos).reshape(-1, 2)
    return float(1.0 / (1.0 + mapa[pts[:, 1], pts[:, 0]].mean()))


def refinar(malha, alvo: np.ndarray | None, quadro: np.ndarray, estado: dict,
            reducao: float = 0.25, orcamento: int = 400,
            focal: float = FOCAL) -> tuple[dict, float]:
    """Ajusta a pose grosseira do usuario ate encaixar na peca.

    Subida de encosta com passo que cai pela metade quando trava. Nao e busca
    global de proposito: o chute do usuario ja poe a peca perto, e busca global
    aqui custaria minutos por causa do custo de renderizar. O que ela faz e o
    ajuste fino que a mao nao consegue - meio grau, cinco pixels.

    A nota mistura borda e, quando ha segmentacao, IoU. So borda tem um otimo
    degenerado (encolher o contorno sobre qualquer aresta forte do fundo); so
    IoU depende de uma mascara que nesta cabine nem sempre existe. Juntas, uma
    segura a outra.

    A busca roda numa versao reduzida do frame: a nota muda pouco com a
    resolucao e o custo cai com o quadrado dela.
    """
    pequeno = cv2.resize(quadro, None, fx=reducao, fy=reducao,
                         interpolation=cv2.INTER_AREA)
    mapa = mapa_de_bordas(pequeno)
    peq = (cv2.resize(alvo, (pequeno.shape[1], pequeno.shape[0]),
                      interpolation=cv2.INTER_NEAREST) if alvo is not None else None)
    # a perspectiva e fixada pela altura inicial e nao acompanha a busca: se ela
    # mudasse a cada passo, o cache de render zeraria e a busca ficaria inviavel.
    # O erro que isso deixa e pequeno porque a busca mexe pouco na altura.
    distancia = distancia_para(estado["altura"], focal)

    cache: dict[tuple, np.ndarray | None] = {}
    gastos = 0

    def render(yaw, pitch, giro, desvio):
        nonlocal gastos
        chave = (round(yaw, 1), round(pitch, 1), round(giro, 1), desvio)
        if chave not in cache:
            cache[chave] = renderizar(malha, *chave[:3], tamanho=RENDER,
                                      distancia=distancia, desvio=desvio)
            gastos += 1
        return cache[chave]

    def nota(e):
        # o desvio acompanha cx/cy: mover a peca no quadro muda de que angulo a
        # camera a ve, e sem isso a busca compensaria isso girando o yaw
        recorte = render(e["yaw"], e["pitch"], e["giro"],
                         desvio_para(e["cx"], e["cy"], quadro.shape, focal))
        if recorte is None:
            return 0.0
        posto = colocar(recorte, mapa.shape, e["cx"] * reducao, e["cy"] * reducao,
                        e["altura"] * reducao, e["espelho"])
        valor = nota_de_bordas(posto, mapa)
        if peq is not None:
            valor = 0.5 * valor + 0.5 * modelo3d.iou(peq, posto)
        return valor

    melhor = dict(estado)
    valor = nota(melhor)

    # o espelho e binario: testa uma vez em vez de deixar para a subida
    virado = dict(melhor, espelho=not melhor["espelho"])
    if nota(virado) > valor:
        melhor, valor = virado, nota(virado)

    passos = dict(PASSOS)
    while gastos < orcamento and passos["yaw"] >= 1.0:
        avancou = False
        for campo, passo in passos.items():
            for sinal in (1, -1):
                tentativa = dict(melhor)
                novo = tentativa[campo] + sinal * passo
                if campo in LIMITES:
                    baixo, alto = LIMITES[campo]
                    novo = max(baixo, min(alto, novo))
                if campo in ("yaw", "giro"):
                    novo %= 360
                tentativa[campo] = novo
                atual = nota(tentativa)
                if atual > valor:
                    melhor, valor, avancou = tentativa, atual, True
                    break
            if gastos >= orcamento:
                break
        if not avancou:
            passos = {k: v / 2 for k, v in passos.items()}

    for campo in ("cx", "cy", "altura"):
        melhor[campo] = int(round(melhor[campo]))
    return melhor, valor


def por_quadrante(malha, alvo: np.ndarray | None, quadro: np.ndarray,
                  estado: dict, sondagem: int = 140,
                  focal: float = FOCAL) -> tuple[dict, float]:
    """Testa as quatro maneiras de pendurar a peca e refina so a que vingou.

    `refinar` sozinho e subida de encosta: partindo do yaw errado ele fica preso
    no maximo local mais proximo, porque para sair dele teria que piorar a nota
    atravessando 90 graus. Aqui os quatro quadrantes sao sementes independentes,
    cada uma com orcamento curto, e so a vencedora paga o refino completo.
    """
    melhor, valor = dict(estado), -1.0
    for volta in (0, 90, 180, 270):
        semente = dict(estado, yaw=(estado["yaw"] + volta) % 360)
        achado, nota = refinar(malha, alvo, quadro, semente,
                               orcamento=sondagem, focal=focal)
        if nota > valor:
            melhor, valor = achado, nota
    return refinar(malha, alvo, quadro, melhor, focal=focal)


AJUDA = [
    "a/d yaw   w/s pitch   z/x giro   +/- tamanho   setas move",
    "e espelha   f encaixa na peca   TAB troca a peca   r volta ao padrao",
    "ENTER refina   o testa os 4 quadrantes   t gira 90   k grava",
    "espaco novo frame   , . navega o dataset   n proximo modelo   ESC sai",
]


def main():
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="cabine", choices=CAMERA_NAMES)
    p.add_argument("--imagem", help="usa uma foto em vez da camera")
    p.add_argument("--modelos", default="modelos")
    p.add_argument("--hooks", help="calibracao especifica")
    p.add_argument("--escala", type=float, default=0.6, help="zoom da janela")
    p.add_argument("--dataset", help="pasta de frames rotulados para navegar com , e .")
    p.add_argument("--poses", default="poses", help="onde gravar as poses confirmadas")
    p.add_argument("--fundo", help="imagem da cabine vazia; padrao local_cv/background_<camera>.jpg")
    p.add_argument("--focal", type=float, default=FOCAL,
                   help="distancia focal em px; 0 desliga. Medida na cabine: "
                        "1012 (ver local_cv/fuga.py)")
    args = p.parse_args()

    # sem fundo a segmentacao cai para as regras de cor, que foram medidas como
    # incapazes de separar peca de piso nesta cabine
    caminho_fundo = args.fundo or f"local_cv/background_{args.camera}.jpg"
    fundo = cv2.imread(caminho_fundo)
    print(f"fundo: {caminho_fundo}" if fundo is not None
          else f"sem fundo em {caminho_fundo} - segmentacao por cor (pior)")

    caminhos = modelos_em(args.modelos)
    if not caminhos:
        raise SystemExit(f"Nenhum modelo 3D em {args.modelos}/")

    # guarda as malhas ja lidas: trocar de modelo e voltar releria o STEP
    # inteiro (segundos), e a malha nao muda enquanto a janela esta aberta
    malhas: dict[str, object] = {}

    def malha_de(caminho: str):
        if caminho not in malhas:
            print(f"carregando {rotulo(caminho)} ...")
            malhas[caminho] = modelo3d.carregar(caminho)
        return malhas[caminho]

    idx = 0
    malha = malha_de(caminhos[idx])

    cap = None
    quadros: list[str] = []
    q_idx = 0
    origem = args.imagem or f"camera {args.camera}"
    if args.dataset:
        quadros = sorted(glob.glob(os.path.join(args.dataset, f"*_{args.camera}.jpg")))
        if not quadros:
            raise SystemExit(f"Nenhum frame de {args.camera} em {args.dataset}/")
        base = cv2.imread(quadros[0])
        origem = os.path.basename(quadros[0])
    elif args.imagem:
        base = cv2.imread(args.imagem)
        if base is None:
            raise SystemExit(f"Nao consegui abrir {args.imagem}")
    else:
        cap = cv2.VideoCapture(camera_url(args.camera), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise SystemExit(f"Nao consegui abrir a camera {args.camera}")
        for _ in range(10):
            ok, base = cap.read()
        if not ok:
            raise SystemExit("Sem frame da camera")

    if args.hooks:
        hooks = load_hooks(args.hooks)
    else:
        _, hooks, _ = pick_calibration(base, args.camera)

    alt, larg = base.shape[:2]
    estado = {**POSE_PADRAO,
              "cx": larg // 2, "cy": alt // 2, "altura": alt // 3,
              "espelho": False}
    inicial = dict(estado)

    arraste = {"ativo": False, "x": 0, "y": 0}
    # instante do ultimo movimento. O render de 320 px custa ~0.26s, e desde que
    # a posicao no quadro entrou na geometria (desvio) e o zoom tambem
    # (distancia), arrastar passou a re-renderizar: a cada ~70 px o desvio muda
    # de degrau. Atravessar o quadro dava uns 27 renders seguidos, ou seja 7s de
    # janela travada. Enquanto a mao esta mexendo, a peca segue o mouse com o
    # render antigo - que so erra a FORMA, nao a posicao - e a geometria e
    # recalculada quando a mao para.
    mexeu = {"em": 0.0}

    def mouse(evento, x, y, flags, _):
        gx, gy = int(x / args.escala), int(y / args.escala)
        if evento == cv2.EVENT_LBUTTONDOWN:
            arraste.update(ativo=True, x=gx, y=gy)
        elif evento == cv2.EVENT_MOUSEMOVE and arraste["ativo"]:
            estado["cx"] += gx - arraste["x"]
            estado["cy"] += gy - arraste["y"]
            arraste.update(x=gx, y=gy)
            mexeu["em"] = time.time()
        elif evento == cv2.EVENT_LBUTTONUP:
            arraste["ativo"] = False
        elif evento == cv2.EVENT_MOUSEWHEEL:
            estado["altura"] = max(20, estado["altura"] + (20 if flags > 0 else -20))
            mexeu["em"] = time.time()

    janela = f"3D sobre a camera {label(args.camera)}"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, int(larg * args.escala), int(alt * args.escala))
    cv2.setMouseCallback(janela, mouse)

    def analisar(img):
        """O que muda quando o frame muda. O mapa de bordas custa caro para
        recalcular a cada iteracao do laco, entao so sai daqui."""
        return (silhuetas(img, hooks, fundo=fundo) if hooks else []), mapa_de_bordas(img)

    def gravar():
        """Grava um exemplo completo: frame cru, vista anotada e pose.

        O frame CRU e a parte que importa. Uma pose sem o frame que a gerou nao
        serve para nada depois - aconteceu nesta sessao: havia uma pose otima do
        Stick e nao deu para reavaliar nada com ela porque so a imagem ja
        desenhada tinha sido salva. Com o par (frame, pose) da para medir
        qualquer mudanca futura contra o que o olho ja aprovou.

        O carimbo de tempo no nome e de proposito: gravar duas vezes a mesma
        peca antes sobrescrevia o arquivo e a base nunca crescia.
        """
        carimbo = time.strftime("%Y%m%d-%H%M%S")
        nome = f"{carimbo}_{rotulo(caminhos[idx])}_{args.camera}.jpg"
        cru = os.path.join(args.poses, "frames", nome)
        vista = os.path.join(args.poses, "vistas", nome)
        os.makedirs(os.path.dirname(cru), exist_ok=True)
        os.makedirs(os.path.dirname(vista), exist_ok=True)
        cv2.imwrite(cru, base)
        cv2.imwrite(vista, frame)

        destino = os.path.join(args.poses, f"{rotulo(caminhos[idx])}.jsonl")
        with open(destino, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "carimbo": carimbo,
                "frame": cru.replace("\\", "/"),
                "origem": origem, "camera": args.camera,
                "modelo": os.path.basename(caminhos[idx]),
                "caixa": pecas[alvo_idx]["caixa"] if pecas else None,
                "modo_segmentacao": pecas[alvo_idx]["modo"] if pecas else None,
                "iou": round(casamento, 4), "bordas": round(nota_borda, 4),
                **estado,
            }) + "\n")
        print(f"gravado {nome}  (IoU {casamento:.3f}, bordas {nota_borda:.3f})"
              f" -> {destino}")

    pausado = cap is None
    # o render sombreado custa ~0.8s; sem cache cada tecla de rotacao trava a
    # janela, e voltar num angulo ja visto e comum ao procurar a pose
    vistas: dict[tuple, tuple] = {}

    def vista_de(yaw, pitch, giro, altura, cx, cy):
        # a distancia entra na chave porque muda a FORMA do render, nao so a
        # escala; o desvio entra porque a peca na borda do quadro mostra outra
        # face que no centro
        distancia = distancia_para(altura, args.focal)
        desvio = desvio_para(cx, cy, base.shape, args.focal)
        chave = (round(yaw, 1), round(pitch, 1), round(giro, 1), distancia, desvio)
        if chave not in vistas:
            vistas[chave] = renderizar_visivel(malha, *chave[:3],
                                               distancia=distancia,
                                               desvio=desvio)
        return vistas[chave], chave

    (render_cor, recorte), chave = vista_de(
        estado["yaw"], estado["pitch"], estado["giro"],
        estado["altura"], estado["cx"], estado["cy"])
    pecas, mapa = analisar(base)
    alvo_idx = 0
    ultimo_refino = ""
    ultima_analise = time.time()

    while True:
        if cap is not None and not pausado:
            ok, novo = cap.read()
            if ok:
                base = novo
                # a segmentacao custa ~0.19s no frame de 1080p: rodar a cada
                # frame derruba a janela para 5 FPS e trava o posicionamento.
                # O que ela produz (contornos das pecas, mapa de bordas) muda
                # devagar, entao meio segundo de atraso nao atrapalha.
                agora = time.time()
                if not arraste["ativo"] and agora - ultima_analise > 0.5:
                    pecas, mapa = analisar(base)
                    alvo_idx = min(alvo_idx, max(0, len(pecas) - 1))
                    ultima_analise = agora

        # so recalcula a geometria quando a mao para: durante o arraste a peca
        # ja segue o mouse com o render anterior, e re-renderizar a cada degrau
        # de desvio travaria a janela
        quieto = not arraste["ativo"] and time.time() - mexeu["em"] > 0.25
        atual = (round(estado["yaw"], 1), round(estado["pitch"], 1),
                 round(estado["giro"], 1),
                 distancia_para(estado["altura"], args.focal),
                 desvio_para(estado["cx"], estado["cy"], base.shape, args.focal))
        if quieto and atual != chave:
            (render_cor, recorte), chave = vista_de(
                estado["yaw"], estado["pitch"], estado["giro"],
                estado["altura"], estado["cx"], estado["cy"])

        frame = base.copy()
        # todas as pecas em contorno fino; a escolhida em contorno grosso, para
        # ficar claro contra qual delas o IoU esta sendo calculado
        for i, peca in enumerate(pecas):
            m = mascara_cheia(peca, frame.shape)
            desenhar(frame, m, COR_CAM, 0.20 if i == alvo_idx else 0.06,
                     3 if i == alvo_idx else 1)
        alvo = mascara_cheia(pecas[alvo_idx], frame.shape) if pecas else None

        sobre = None
        if recorte is not None:
            sobre = colocar(recorte, frame.shape, estado["cx"], estado["cy"],
                            estado["altura"], estado["espelho"])
            pintado = colocar(render_cor, frame.shape, estado["cx"], estado["cy"],
                              estado["altura"], estado["espelho"])
            desenhar_solido(frame, pintado, sobre, 0.75, 2)

        for h in hooks:
            cv2.circle(frame, (int(h["x"]), int(h["y"])), 4, (255, 180, 0), -1)

        casamento = modelo3d.iou(alvo, sobre) if alvo is not None and sobre is not None else 0.0
        nota_borda = nota_de_bordas(sobre, mapa) if sobre is not None else 0.0
        modo_seg = pecas[alvo_idx]["modo"] if pecas else "-"
        painel(frame, [
            f"{rotulo(caminhos[idx])}  ({idx + 1}/{len(caminhos)})"
            f"{'  espelhado' if estado['espelho'] else ''}",
            f"{origem}",
            f"peca {alvo_idx + 1}/{len(pecas)} (seg {modo_seg})" if pecas
            else "nenhuma peca segmentada",
            f"yaw {estado['yaw']:.0f}  pitch {estado['pitch']:.0f}  "
            f"giro {estado['giro']:.0f}  altura {estado['altura']}px",
            f"bordas {nota_borda:.3f}   IoU {casamento:.3f}   {ultimo_refino}",
        ] + AJUDA)

        cv2.imshow(janela, cv2.resize(frame, None, fx=args.escala, fy=args.escala))
        tecla = cv2.waitKey(1 if not pausado else 30) & 0xFF
        if tecla != 255:
            # qualquer tecla adia o render: segurando 'd' para girar, o que
            # interessa e o angulo onde a mao soltou, nao os vinte do caminho
            mexeu["em"] = time.time()

        if tecla in (27, ord("q")):
            break
        elif tecla == ord("a"):
            estado["yaw"] = (estado["yaw"] - 5) % 360
        elif tecla == ord("d"):
            estado["yaw"] = (estado["yaw"] + 5) % 360
        elif tecla == ord("w"):
            estado["pitch"] = min(89.0, estado["pitch"] + 5)
        elif tecla == ord("s"):
            estado["pitch"] = max(-89.0, estado["pitch"] - 5)
        elif tecla == ord("z"):
            estado["giro"] = (estado["giro"] - 5) % 360
        elif tecla == ord("x"):
            estado["giro"] = (estado["giro"] + 5) % 360
        elif tecla in (ord("+"), ord("=")):
            estado["altura"] += 20
        elif tecla in (ord("-"), ord("_")):
            estado["altura"] = max(20, estado["altura"] - 20)
        elif tecla == ord("e"):
            estado["espelho"] = not estado["espelho"]
        elif tecla == 82:
            estado["cy"] -= 10
        elif tecla == 84:
            estado["cy"] += 10
        elif tecla == 81:
            estado["cx"] -= 10
        elif tecla == 83:
            estado["cx"] += 10
        elif tecla == ord("f") and pecas:
            x, y, w, h = pecas[alvo_idx]["caixa"]
            estado.update(cx=x + w // 2, cy=y + h // 2, altura=h)
        elif tecla == 9 and pecas:            # TAB: alterna entre as pecas
            alvo_idx = (alvo_idx + 1) % len(pecas)
        elif tecla in (13, 10):   # ENTER: refina sozinho
            print("refinando ...")
            estado, achado = refinar(malha, alvo, base, estado, focal=args.focal)
            chave = None
            ultimo_refino = f"(nota {achado:.3f})"
            print(f"  nota {achado:.3f}  yaw {estado['yaw']:.0f} "
                  f"pitch {estado['pitch']:.0f} giro {estado['giro']:.0f} "
                  f"altura {estado['altura']}")
        elif tecla == ord("t"):   # muda de lado: 90 graus no eixo vertical (Y do modelo)
            malha.vertices = np.asarray(malha.vertices) @ VIRA_LADO.T
            vistas.clear()   # a malha girou: todo render em cache virou invalido
            chave = None
        elif tecla == ord("o"):   # testa os quatro lados e refina o melhor
            print("testando os 4 quadrantes ...")
            estado, achado = por_quadrante(malha, alvo, base, estado,
                                           focal=args.focal)
            chave = None
            ultimo_refino = f"(nota {achado:.3f})"
            print(f"  nota {achado:.3f}  yaw {estado['yaw']:.0f} "
                  f"pitch {estado['pitch']:.0f} giro {estado['giro']:.0f} "
                  f"altura {estado['altura']}")
        elif tecla in (ord("k"), ord("g")):
            gravar()
        elif tecla == ord("r"):
            estado.update(inicial)
        elif tecla == ord("n"):
            idx = (idx + 1) % len(caminhos)
            malha = malha_de(caminhos[idx])
            vistas.clear()   # o cache e por angulo, nao por modelo
            chave = None
        elif tecla == ord("p"):
            pausado = not pausado
        elif tecla == ord(" ") and cap is not None:
            ok, novo = cap.read()
            if ok:
                base = novo
                pecas, mapa = analisar(base)
                alvo_idx = 0
        elif tecla in (ord(","), ord(".")) and quadros:
            q_idx = (q_idx + (1 if tecla == ord(".") else -1)) % len(quadros)
            base = cv2.imread(quadros[q_idx])
            origem = os.path.basename(quadros[q_idx])
            pecas, mapa = analisar(base)
            alvo_idx = 0
            ultimo_refino = ""
            print(f"{q_idx + 1}/{len(quadros)}  {origem}  {len(pecas)} peca(s)")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
