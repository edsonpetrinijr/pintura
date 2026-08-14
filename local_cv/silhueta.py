"""Silhueta das pecas penduradas na cabine.

O caminho principal e SUBTRACAO DE FUNDO (mascara_fundo): a cabine e fixa, e o
que muda entre um frame e outro e exatamente o que esta pendurado.

Cor nao serve, e isso foi medido, nao suposto. Em tres frames da mesma camera no
mesmo dia a peca apareceu escura (aco cru), amarela (recem pintada) e branca
(fundida crua). E no frame amarelo a saturacao da peca era 148 - a mesma do piso
(152) e a das capas de robo (148). A matiz nao separa nada porque a cabine
inteira e amarela de overspray: peca 25, capa 24 a 28, piso 26.

As duas mascaras de cor (mascara_escura, mascara_pintada) ficam como reserva
para quando nao ha fundo montado, com os limiares que funcionaram nos frames em
que cada uma foi calibrada. Nao espere que generalizem.

Duas armadilhas que este modulo trata:

1. Um operador passando na frente parte a peca em varios blobs. Por isso os
   blobs proximos sao unidos antes de virar peca.
2. A lente e olho de peixe e escurece os cantos do frame. Esses cantos viram
   blob escuro gigante. Por isso blobs encostados na borda sao descartados - mas
   so no modo escuro, que e onde o problema aparece.
"""
import cv2
import numpy as np

V_MAX = 90       # ate onde e "escuro"
S_MAX = 110      # acima disso e amarelo da cabine, nao peca
S_MIN_PINTADA = 190   # peca pintada mede S=227; piso e capas ficam em 145-164
V_MIN_PINTADA = 60    # abaixo disso e piso/grade na sombra
V_MAX_PINTADA = 180   # acima disso e capa de robo iluminada
AREA_MIN = 4000  # px; abaixo disso e mangueira, sombra, sujeira
FOLGA_BORDA = 12  # px de tolerancia para considerar que encostou na borda
# Vao abaixo do qual dois blobs viram a mesma peca. Depende da mascara: a
# escura passa por um fechamento de 35 px e tolera vao maior; a pintada fecha
# com 15 px. E ha um teto medido: duas pecas VIZINHAS na linha ficaram a 82 px
# uma da outra, entao unir a 90 juntava peca distinta.
VAO_UNIAO = {"escura": 90, "pintada": 60, "fundo": 60}


def _limpar(mask: np.ndarray, y_min: int, abertura: int, fechamento: int) -> np.ndarray:
    if y_min > 0:
        mask[:y_min, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((abertura, abertura), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((fechamento, fechamento), np.uint8))


def mascara_escura(frame, v_max: int = V_MAX, s_max: int = S_MAX,
                   y_min: int = 0) -> np.ndarray:
    """Mascara do que e escuro e pouco saturado, abaixo da linha y_min.

    Serve para a peca ainda sem tinta.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    return _limpar(((v < v_max) & (s < s_max)).astype(np.uint8) * 255, y_min, 9, 35)


def mascara_pintada(frame, s_min: int = S_MIN_PINTADA, v_min: int = V_MIN_PINTADA,
                    v_max: int = V_MAX_PINTADA, y_min: int = 0) -> np.ndarray:
    """Mascara da peca ja pintada: amarelo muito saturado, brilho medio."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    bruta = ((s > s_min) & (v > v_min) & (v < v_max)).astype(np.uint8) * 255
    return _limpar(bruta, y_min, 5, 15)


LIMIAR_FUNDO = 40  # px de diferenca contra o fundo; medido: p90 do frame e 33


def mascara_fundo(frame, fundo, limiar: int = LIMIAR_FUNDO) -> np.ndarray:
    """O que mudou em relacao a cabine vazia. Nao olha cor nenhuma.

    Este e o caminho certo, e as duas mascaras de cor acima ficam como reserva
    para quando nao ha fundo montado. Medido em tres frames da mesma camera no
    mesmo dia, a peca apareceu ESCURA (aco cru), AMARELA (recem pintada) e
    BRANCA (fundida crua) - e no frame amarelo a saturacao dela (148) era
    identica a do piso (152) e a das capas de robo (148). Nenhum limiar de cor
    sobrevive a isso. Ja a cabine e fixa: o que muda de um frame para outro e
    exatamente o que esta pendurado.

    Nao aplica y_min: a peca sobe ate a altura dos ganchos (o Tandem medido
    comeca em y=374, acima da linha y_min=463 tirada da calibracao), e cortar
    por altura decapitaria a peca. O ruido de cima sao as lampadas piscando, e
    elas caem sozinhas no filtro de area.
    """
    diferenca = cv2.absdiff(frame, fundo).max(axis=2)
    bruta = (diferenca > limiar).astype(np.uint8) * 255
    return _limpar(bruta, 0, 7, 25)


def mascara_corrente(frame, fundo, limiar: int = LIMIAR_FUNDO) -> np.ndarray:
    """Subtracao de fundo SEM abertura: preserva a CORRENTE fina do gancho.

    mascara_fundo abre com 7 px para tirar ruido, o que e certo quando o alvo e
    o BLOB da peca - mas a corrente tem ~3-5 px de largura e some nessa abertura.
    Para OCUPACAO por gancho o sinal e justamente a corrente descendo do gancho,
    entao aqui so ha um fechamento leve para ligar os elos, sem abertura.

    Medido (logs/validacao_ganchos.csv, 53 frames, ganchos que a API confirma
    ocupados): a cobertura desta mascara numa janela abaixo do gancho recupera
    0.64 deles contra 0.48 das bordas no mesmo nivel de falso positivo, enquanto
    mascara_fundo (com abertura) cai para 0.39 porque apaga a corrente.
    """
    diferenca = cv2.absdiff(frame, fundo).max(axis=2)
    bruta = (diferenca > limiar).astype(np.uint8) * 255
    return cv2.morphologyEx(bruta, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def _maior_blob_valido(mask, shape) -> int:
    """Area do maior blob que nao encosta na borda, para comparar mascaras."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)
             if not _encosta_na_borda(
                 (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                  stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]), shape)]
    return max(areas, default=0)


def mascara_peca(frame, y_min: int = 0, modo: str = "auto",
                 fundo=None) -> tuple[np.ndarray, str]:
    """Escolhe entre peca crua e peca pintada, e devolve a mascara e o modo usado.

    Com fundo disponivel ele ganha sempre: nao depende de cor, e cor foi medida
    como incapaz de separar peca de piso nesta cabine.

    Sem fundo, sobra escolher entre as duas regras de cor. Nao une as duas de
    proposito. Unir contamina: no frame medido, a regra de escuro pegava um blob
    de 16 mil px colado na blade pintada, e a uniao esticava a caixa da peca ate
    ele. Escolher a mascara com o maior blob valido acerta porque a peca domina
    o frame na etapa em que aparece.
    """
    if fundo is not None and modo in ("auto", "fundo"):
        return mascara_fundo(frame, fundo), "fundo"
    if modo == "escura":
        return mascara_escura(frame, y_min=y_min), "escura"
    if modo == "pintada":
        return mascara_pintada(frame, y_min=y_min), "pintada"

    escura = mascara_escura(frame, y_min=y_min)
    pintada = mascara_pintada(frame, y_min=y_min)
    if _maior_blob_valido(pintada, frame.shape) >= _maior_blob_valido(escura, frame.shape):
        return pintada, "pintada"
    return escura, "escura"


def _encosta_na_borda(caixa, shape) -> bool:
    x, y, w, h = caixa
    alt, larg = shape[:2]
    return (x <= FOLGA_BORDA or y <= FOLGA_BORDA
            or x + w >= larg - FOLGA_BORDA or y + h >= alt - FOLGA_BORDA)


def _unir(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    return (x, y, max(a[0] + a[2], b[0] + b[2]) - x,
            max(a[1] + a[3], b[1] + b[3]) - y)


def _perto(a, b, vao: int) -> bool:
    """Caixas separadas por menos de `vao` px nos dois eixos."""
    dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
    return dx <= vao and dy <= vao


def _agrupar(caixas, vao: int):
    """Une caixas vizinhas ate estabilizar (operador partindo a peca em duas)."""
    grupos = list(caixas)
    mudou = True
    while mudou:
        mudou = False
        for i in range(len(grupos)):
            for j in range(i + 1, len(grupos)):
                if _perto(grupos[i], grupos[j], vao):
                    grupos[i] = _unir(grupos[i], grupos[j])
                    del grupos[j]
                    mudou = True
                    break
            if mudou:
                break
    return grupos


def descritores(mask_recorte: np.ndarray) -> dict:
    """Medidas de forma invariantes a escala, para comparar com o modelo 3D."""
    alt, larg = mask_recorte.shape[:2]
    area = float(np.count_nonzero(mask_recorte))
    hu = cv2.HuMoments(cv2.moments(mask_recorte)).flatten()
    # log dos momentos de Hu: sem isso a escala deles varia ordens de grandeza
    hu_log = [float(-np.sign(h) * np.log10(abs(h))) if h else 0.0 for h in hu]
    return {
        "razao": round(larg / alt, 3) if alt else 0.0,
        "preenchimento": round(area / (larg * alt), 3) if larg and alt else 0.0,
        "area": int(area),
        "hu": [round(h, 3) for h in hu_log],
    }


def silhuetas(frame, hooks: list[dict], area_min: int = AREA_MIN,
              vao_uniao: int | None = None, modo: str = "auto",
              fundo=None) -> list[dict]:
    """Pecas penduradas, com caixa, mascara e descritores de forma.

    Nao devolve a quais ganchos cada peca pertence, de proposito: na vista da
    cabine a perspectiva comprime os ganchos, e a caixa de uma peca so cobre o
    x de nove deles (medido). Quem sabe os ganchos e a API; aqui a entrega e a
    forma.
    """
    base = max((h["y"] for h in hooks), default=0) + 40
    mask, usado = mascara_peca(frame, y_min=base, modo=modo, fundo=fundo)

    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    caixas = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < area_min:
            continue
        caixa = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                 stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        # so a mascara escura sofre com a borda: o vinhetamento do olho de peixe
        # escurece os cantos e vira blob gigante. Na mascara pintada isso nao
        # acontece, e descartar borda ali custa caro - medido, jogava fora uma
        # blade inteira (272x438) so porque ela chega no fim do frame.
        if usado == "escura" and _encosta_na_borda(caixa, frame.shape):
            continue
        caixas.append(caixa)

    achados = []
    for x, y, w, h in _agrupar(caixas, VAO_UNIAO[usado] if vao_uniao is None else vao_uniao):
        recorte = mask[y:y + h, x:x + w]
        if np.count_nonzero(recorte) < area_min:
            continue
        achados.append({
            "caixa": (int(x), int(y), int(w), int(h)),
            "mask": recorte,
            "modo": usado,
            **descritores(recorte),
        })

    achados.sort(key=lambda p: -p["area"])
    return achados
