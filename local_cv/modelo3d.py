"""Modelo 3D (STEP) -> banco de silhuetas para casar com o que a camera ve.

A ideia e nao precisar de dataset. Nao ha fotos suficientes das pecas (2 a 6 por
tipo, sem caixa delimitadora), mas ha o CAD. Renderizando o CAD de muitos
angulos e comparando a silhueta com a que sai de local_cv/silhueta.py, da para
identificar a peca sem treinar nada.

A renderizacao aqui e feita na mao com cv2.fillPoly, e nao com OpenGL/pyrender.
Motivo: silhueta nao precisa de luz, material nem profundidade, so da uniao dos
triangulos projetados - e assim nao entra dependencia de contexto grafico, que
e o que costuma quebrar em maquina sem GPU e em servico headless.

Nao ha correcao de distorcao de lente, mas a projecao E em perspectiva. Foi
medido nesta cabine que ela importa: o Tandem tem 2.2 m e ocupa 676 px de 1920,
entao subtende cerca de 31 graus - longe do limite ortografico. A forca da
perspectiva entra como DISTANCIA, em raios da peca, e sai da distancia focal
da camera via `distancia_em_raios = 2 * f / altura_em_px` (ver local_cv/fuga.py,
que estima f pelos pontos de fuga da cabine).
"""
import hashlib
import json
import os

import cv2
import numpy as np

# Subpasta oculta de proposito: os GLB convertidos e os bancos ficam junto dos
# modelos, mas fora do alcance de quem varre a pasta procurando CAD. Sem isso o
# GLB de cache do 4175193 vira um "modelo" novo e empata com o original.
CACHE = os.path.join("modelos", ".cache")
TAMANHO = 256      # lado do quadro de silhueta normalizada
MARGEM = 0.06      # folga relativa dentro do quadro
# Faixa de pitch do banco. Comecou em +-20 supondo que a peca so balanca, mas
# ajuste manual sobre um frame real fechou em pitch 44: a camera olha a peca de
# cima, entao a inclinacao aparente e grande mesmo com a peca parada.
# Nao adianta decimar a malha para acelerar: medido, o STEP e uma montagem de
# muitos corpos e a decimacao quadratica colapsa os corpos pequenos e rasga as
# chapas (preenchimento cai de 34% para 23% com 40k faces). O custo se paga uma
# vez so por modelo via banco_cacheado.
PITCHES = (-45, -30, -15, 0, 15, 30, 45)
# Distancia da camera ao centro da peca, em raios da peca. Nao e chute: o Tandem
# (2.2 m, raio 1.4 m) aparece com 676 px de largura num frame de 1920, e com f
# da ordem de 950 px isso da d = 2*950/676 * raio ~ 2.8 raios. Menor que isso a
# peca deforma; acima de ~20 vira ortografico de novo.
DISTANCIA = 3.0


def _glb_de(step_path: str, cache_dir: str) -> str:
    """Converte STEP para GLB uma vez so, com cache pelo conteudo do arquivo."""
    import cascadio

    os.makedirs(cache_dir, exist_ok=True)
    with open(step_path, "rb") as f:
        # So o inicio: STEP de peca tem dezenas de MB e o cabecalho ja carrega
        # nome, data e sistema de origem, o bastante para detectar troca.
        digest = hashlib.sha256(f.read(1 << 20)).hexdigest()[:16]

    destino = os.path.join(cache_dir, f"{digest}.glb")
    if not os.path.exists(destino):
        cascadio.step_to_glb(step_path, destino, tol_linear=0.1, tol_angular=0.5)
    return destino


def carregar(caminho: str, cache_dir: str = CACHE):
    """Le STEP/STP (ou GLB/STL direto) e devolve uma malha unica em metros."""
    import trimesh

    if caminho.lower().endswith((".step", ".stp")):
        caminho = _glb_de(caminho, cache_dir)

    carregado = trimesh.load(caminho)
    if isinstance(carregado, trimesh.Scene):
        # dump aplica as transformacoes do grafo da cena. Concatenar as
        # geometrias cruas ignora essas transformacoes e a peca sai do tamanho
        # e da posicao errados.
        carregado = carregado.dump(concatenate=True)
    return carregado


def _rotacao(yaw: float, pitch: float) -> np.ndarray:
    """Matriz que leva o mundo para a vista, em graus."""
    a, b = np.radians(yaw), np.radians(pitch)
    rz = np.array([[np.cos(a), -np.sin(a), 0],
                   [np.sin(a), np.cos(a), 0],
                   [0, 0, 1.0]])
    rx = np.array([[1.0, 0, 0],
                   [0, np.cos(b), -np.sin(b)],
                   [0, np.sin(b), np.cos(b)]])
    return rx @ rz


def _fora_do_eixo(desvio) -> np.ndarray:
    """Rotacao que corresponde a ver a peca fora do centro da imagem.

    Camera de 87 graus de abertura nao ve todo mundo do mesmo jeito: uma peca na
    borda direita do frame esta a mais de 40 graus do eixo optico e mostra a
    face ESQUERDA dela, enquanto a mesma peca no centro mostra a face de frente.
    Sem isso o render so acerta a pose no meio do quadro, e o refinador
    compensa a diferenca torcendo o yaw - ou seja, grava uma pose errada.

    `desvio` sao os angulos (horizontal, vertical) em graus da peca em relacao
    ao eixo optico. A rotacao devolvida e a menor que leva a direcao real de
    visada de volta ao eixo, para que o resto do pipeline continue renderizando
    como se estivesse centrado.
    """
    ax, ay = np.radians(desvio)
    # eixo de vista aponta para -z; y da imagem cresce para baixo, entao inverte
    d = np.array([np.tan(ax), -np.tan(ay), -1.0])
    d /= np.linalg.norm(d)
    alvo = np.array([0.0, 0.0, -1.0])

    v = np.cross(d, alvo)
    seno = np.linalg.norm(v)
    if seno < 1e-12:          # ja esta no eixo
        return np.eye(3)
    cosseno = float(d @ alvo)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - cosseno) / seno ** 2)


def _projetar(malha, yaw: float, pitch: float, tamanho: int,
              distancia: float = DISTANCIA, desvio=None):
    """Vertices no quadro da imagem, os vertices na vista e a posicao do olho.

    Preserva a razao largura/altura: ela e justamente uma das informacoes que
    distinguem as pecas (um Boom e comprido, um Moldboard e chapa larga).

    A divisao por profundidade vem antes do enquadramento, entao a escala final
    continua sendo "cabe no quadro": `distancia` muda a FORMA (o que esta perto
    cresce), nao o tamanho. Com distancia None a projecao volta a ser
    ortografica, que e o caso limite util para conferir se algo mudou de fato.

    `desvio` entra DEPOIS da pose porque e propriedade de onde a peca esta no
    quadro, nao de como ela foi pendurada: girar a peca no gancho nao muda o
    angulo com que a camera a ve.
    """
    R = _rotacao(yaw, pitch)
    if desvio is not None and any(desvio):
        R = _fora_do_eixo(desvio) @ R
    vistos = np.asarray(malha.vertices) @ R.T
    baixo, alto = vistos.min(axis=0), vistos.max(axis=0)
    centro = (baixo + alto) / 2.0
    raio = float(np.linalg.norm(alto - baixo) / 2.0) or 1e-9

    xy = vistos[:, :2] - centro[:2]
    if distancia:
        # z maior = mais perto da camera, entao o olho fica acima do topo em z
        olho = np.array([centro[0], centro[1], centro[2] + distancia * raio])
        # numerador = profundidade do centro: no centro o fator vale 1 e a peca
        # nao muda de tamanho, so de forma
        fundura = np.maximum(olho[2] - vistos[:, 2:3], 1e-6 * raio)
        xy = xy * (distancia * raio / fundura)
    else:
        olho = None

    xy = xy.copy()
    xy[:, 1] *= -1.0  # y da imagem cresce para baixo

    minimo, maximo = xy.min(axis=0), xy.max(axis=0)
    extensao = np.maximum(maximo - minimo, 1e-9)
    util = tamanho * (1 - 2 * MARGEM)
    escala = util / extensao.max()

    pts = (xy - minimo) * escala
    pts += (tamanho - extensao * escala) / 2.0
    return pts, vistos, olho


def silhueta(malha, yaw: float = 0.0, pitch: float = 0.0,
             tamanho: int = TAMANHO, distancia: float = DISTANCIA,
             desvio=None) -> np.ndarray:
    """Silhueta binaria da malha vista de (yaw, pitch), normalizada no quadro."""
    pts, _, _ = _projetar(malha, yaw, pitch, tamanho, distancia, desvio)

    quadro = np.zeros((tamanho + 2, tamanho + 2), np.uint8)
    tri = np.round(pts[np.asarray(malha.faces)]).astype(np.int32) + 1

    # Nao da para usar fillPoly com todos os triangulos de uma vez: ela aplica
    # regra par-impar entre os poligonos da mesma chamada, entao onde os
    # triangulos se sobrepoem o preenchimento se CANCELA e a silhueta sai cheia
    # de furos. Preencher um a um seria correto, mas sao centenas de milhares.
    # Entao: desenha so as arestas, inunda o lado de fora e inverte. A borda de
    # 1 px garante um ponto de partida fora da peca.
    cv2.polylines(quadro, tri, True, 255, 1, cv2.LINE_8)
    fora = quadro.copy()
    cv2.floodFill(fora, np.zeros((tamanho + 4, tamanho + 4), np.uint8), (0, 0), 255)
    return (quadro | ~fora)[1:-1, 1:-1]


def sombreado(malha, yaw: float = 0.0, pitch: float = 0.0,
              tamanho: int = TAMANHO, cor=(70, 210, 255),
              faixas: int = 64, niveis: int = 10,
              descartar_verso: bool = True, distancia: float = DISTANCIA,
              desvio=None):
    """Render solido da malha, sem OpenGL. Devolve (imagem BGR, mascara).

    Algoritmo do pintor: ordena as faces da mais distante para a mais proxima e
    desenha nessa ordem, entao a face da frente cobre a de tras. Nao ha z-buffer
    por pixel, mas para peca fabricada isso quase nao aparece.

    O detalhe que faz funcionar: as faces sao desenhadas em blocos por
    profundidade. Uma unica chamada de fillPoly com a malha toda aplicaria regra
    par-impar e o casco da frente CANCELARIA o de tras. Dentro de uma faixa fina
    de profundidade os triangulos ladrilham a superficie sem se sobrepor, e o
    problema desaparece.

    O tom vem da normal da face contra o eixo da camera: e iluminacao frontal,
    que e o que a cabine tem.
    """
    pts, vistos, olho = _projetar(malha, yaw, pitch, tamanho, distancia, desvio)
    faces = np.asarray(malha.faces)

    borda = vistos[faces[:, 1]] - vistos[faces[:, 0]]
    outra = vistos[faces[:, 2]] - vistos[faces[:, 0]]
    normais = np.cross(borda, outra)

    # As faces de tras seriam desenhadas primeiro e cobertas pelas da frente:
    # so custam tempo. A malha tem winding consistente (verificado), entao o
    # sinal da normal contra a direcao de vista separa as duas metades.
    # Em perspectiva essa direcao muda de face para face - testar so o z, como
    # no caso ortografico, descarta faces visiveis na periferia e abre buraco.
    if descartar_verso:
        if olho is None:
            frente = normais[:, 2] > 0
        else:
            para_olho = olho - vistos[faces].mean(axis=1)
            frente = (normais * para_olho).sum(axis=1) > 0
        if frente.any():
            faces, normais = faces[frente], normais[frente]

    tri = np.round(pts[faces]).astype(np.int32)

    comprimento = np.linalg.norm(normais, axis=1)
    comprimento[comprimento == 0] = 1.0
    luz = np.abs(normais[:, 2] / comprimento)
    nivel = np.clip((luz * niveis).astype(np.int32), 0, niveis - 1)

    profundidade = vistos[faces][:, :, 2].mean(axis=1)
    ordem = np.argsort(profundidade)  # z maior = mais perto da camera

    quadro = np.zeros((tamanho, tamanho, 3), np.uint8)
    bloco = max(1, len(ordem) // faixas)
    for i in range(0, len(ordem), bloco):
        fatia = ordem[i:i + bloco]
        for nv in range(niveis):
            sel = fatia[nivel[fatia] == nv]
            if not len(sel):
                continue
            tom = 0.30 + 0.70 * (nv + 0.5) / niveis
            cv2.fillPoly(quadro, tri[sel],
                         tuple(int(c * tom) for c in cor), cv2.LINE_8)

    # A mascara sai do proprio render: o tom minimo e 0.30, entao todo pixel
    # coberto por algum triangulo e diferente de zero. Chamar silhueta() aqui
    # custaria mais que o render inteiro (polylines sobre 175k triangulos), e
    # daria o mesmo resultado - ela tambem fecha os furos passantes.
    pintado = np.zeros((tamanho + 2, tamanho + 2), np.uint8)
    pintado[1:-1, 1:-1] = (quadro.sum(axis=2) > 0).astype(np.uint8) * 255
    fora = pintado.copy()
    cv2.floodFill(fora, np.zeros((tamanho + 4, tamanho + 4), np.uint8), (0, 0), 255)
    mask = (pintado | ~fora)[1:-1, 1:-1]

    # onde a mascara diz que ha peca mas o pintor deixou buraco (faces
    # coincidentes que se cancelaram na mesma faixa), preenche com tom medio
    vazio = (mask > 0) & (quadro.sum(axis=2) == 0)
    quadro[vazio] = tuple(int(c * 0.55) for c in cor)
    return quadro, mask


def normalizar(mask: np.ndarray, tamanho: int = TAMANHO) -> np.ndarray:
    """Poe uma mascara qualquer no mesmo quadro das silhuetas renderizadas."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros((tamanho, tamanho), np.uint8)

    recorte = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alt, larg = recorte.shape
    util = tamanho * (1 - 2 * MARGEM)
    escala = util / max(alt, larg)
    nova = cv2.resize(recorte, (max(1, int(larg * escala)), max(1, int(alt * escala))),
                      interpolation=cv2.INTER_NEAREST)

    quadro = np.zeros((tamanho, tamanho), np.uint8)
    y0 = (tamanho - nova.shape[0]) // 2
    x0 = (tamanho - nova.shape[1]) // 2
    quadro[y0:y0 + nova.shape[0], x0:x0 + nova.shape[1]] = nova
    return quadro


def banco(malha, passo_yaw: int = 10, pitches=PITCHES,
          tamanho: int = TAMANHO) -> list[dict]:
    """Silhuetas da peca em muitos angulos.

    A peca fica pendurada e gira em torno do proprio eixo vertical, entao o yaw
    e varrido inteiro. O pitch varia pouco: a camera esta fixa e a peca so
    balanca.
    """
    return [{"yaw": float(yaw), "pitch": float(pitch),
             "mask": silhueta(malha, yaw, pitch, tamanho)}
            for yaw in range(0, 360, passo_yaw) for pitch in pitches]


def _digest(caminho: str) -> str:
    with open(caminho, "rb") as f:
        return hashlib.sha256(f.read(1 << 20)).hexdigest()[:16]


def banco_cacheado(caminho: str, passo_yaw: int = 10,
                   pitches=PITCHES, tamanho: int = TAMANHO,
                   cache_dir: str = CACHE) -> list[dict]:
    """Igual a banco(), mas guarda o resultado em disco.

    Renderizar custa caro (medido: 44 a 64 s para 72 vistas de uma peca de 300k
    faces), e o banco so muda se o CAD ou os angulos mudarem. As mascaras vao
    empacotadas em bits: sao binarias, e sem isso o arquivo fica 8x maior.
    """
    os.makedirs(cache_dir, exist_ok=True)
    chave = json.dumps([_digest(caminho), passo_yaw, list(pitches), tamanho],
                       sort_keys=True)
    nome = hashlib.sha256(chave.encode()).hexdigest()[:16]
    destino = os.path.join(cache_dir, f"banco_{nome}.npz")

    if os.path.exists(destino):
        dados = np.load(destino)
        bits, angulos = dados["bits"], dados["angulos"]
        return [{"yaw": float(a[0]), "pitch": float(a[1]),
                 "mask": np.unpackbits(b).reshape(tamanho, tamanho) * 255}
                for b, a in zip(bits, angulos)]

    vistas = banco(carregar(caminho, cache_dir), passo_yaw, pitches, tamanho)
    np.savez_compressed(
        destino,
        bits=np.stack([np.packbits(v["mask"] > 0) for v in vistas]),
        angulos=np.array([[v["yaw"], v["pitch"]] for v in vistas], np.float32))
    return vistas


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ba, bb = a > 0, b > 0
    uniao = np.count_nonzero(ba | bb)
    return float(np.count_nonzero(ba & bb) / uniao) if uniao else 0.0


def casar(mask_camera: np.ndarray, bancos: dict[str, list[dict]],
          tamanho: int = TAMANHO) -> list[dict]:
    """Ordena os modelos pela melhor silhueta que cada um consegue oferecer.

    Devolve um item por modelo, do melhor para o pior, com o angulo que deu o
    melhor encaixe. A diferenca entre o primeiro e o segundo e o que diz se a
    resposta e confiavel - peca simetrica casa bem com varios modelos.
    """
    alvo = normalizar(mask_camera, tamanho)
    espelho = cv2.flip(alvo, 1)  # a peca pode estar virada para o outro lado

    postos = []
    for nome, vistas in bancos.items():
        melhor = max(
            ((max(iou(alvo, v["mask"]), iou(espelho, v["mask"])), v) for v in vistas),
            key=lambda p: p[0])
        postos.append({"modelo": nome, "iou": round(melhor[0], 4),
                       "yaw": melhor[1]["yaw"], "pitch": melhor[1]["pitch"]})

    postos.sort(key=lambda p: -p["iou"])
    for i, p in enumerate(postos):
        p["vantagem"] = round(p["iou"] - postos[i + 1]["iou"], 4) if i + 1 < len(postos) else None
    return postos
