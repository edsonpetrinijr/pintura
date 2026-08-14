"""Estima a geometria da camera a partir dos pontos de fuga da propria cabine.

Por que isso existe: o render 3D precisa saber COMO a camera projeta, e ate aqui
isso era chute. A cabine e um paralelepipedo cheio de aresta reta - trilho,
grade do piso, quina de parede, perfil da estrutura. Tres familias de retas
paralelas e ortogonais entre si, e cada familia converge para um ponto de fuga.
Desses pontos saem a distancia focal e a orientacao da camera sem tabua de
calibracao, sem mexer no equipamento e sem parar a linha.

O que sai daqui alimenta duas coisas:
  - `modelo3d.DISTANCIA`, via `distancia_em_raios = 2 * f / altura_em_px`, que e
    o que tira o render do limite ortografico;
  - a semente de pitch/yaw da pose, que hoje e um numero medido na mao num
    frame so.

Limite conhecido: sem tabua nao da para separar distorcao radial da lente, e
camera de vigilancia costuma ter barril forte nas bordas. As retas usadas aqui
sao filtradas por comprimento, o que ja favorece as centrais, mas se a estimativa
de f ficar instavel entre frames a lente e a primeira suspeita.
"""
import argparse
import glob
import os

import cv2
import numpy as np

COMPRIMENTO_MIN = 60      # px; abaixo disso a direcao da reta vira ruido
TOLERANCIA = np.radians(2.0)
MIN_APOIOS = 12           # retas que uma familia precisa para virar ponto de fuga
SEPARACAO = np.radians(30.0)   # angulo minimo entre duas familias aceitas


def segmentos(frame: np.ndarray, comprimento_min: int = COMPRIMENTO_MIN) -> np.ndarray:
    """Retas longas da imagem, em (x1, y1, x2, y2).

    Bilateral antes do Canny pelo mesmo motivo de mapa_de_bordas: a grade do
    piso e a fumaca geram borda fina em todo canto e sem suavizar preservando
    aresta o detector devolve confete em vez de estrutura.
    """
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(cv2.bilateralFilter(cinza, 7, 60, 60), 50, 150)
    achados = cv2.HoughLinesP(bordas, 1, np.pi / 360, threshold=60,
                              minLineLength=comprimento_min, maxLineGap=6)
    return np.empty((0, 4)) if achados is None else achados.reshape(-1, 4).astype(float)


def _erro(segs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Angulo entre cada segmento e a direcao dele ate o ponto de fuga.

    Em coordenada homogenea para aguentar ponto de fuga no infinito, que e o
    caso normal de familia paralela ao plano da imagem: a terceira coordenada
    vai a zero e a divisao explodiria.
    """
    meio = (segs[:, :2] + segs[:, 2:]) / 2.0
    direcao = segs[:, 2:] - segs[:, :2]
    # vp[2]*meio e o termo que some quando o vp esta no infinito
    ate_vp = vp[:2] - vp[2] * meio
    cruz = np.abs(direcao[:, 0] * ate_vp[:, 1] - direcao[:, 1] * ate_vp[:, 0])
    normas = np.linalg.norm(direcao, axis=1) * np.linalg.norm(ate_vp, axis=1)
    return np.arcsin(np.clip(cruz / np.maximum(normas, 1e-9), 0, 1))


def _um_vp(segs: np.ndarray, tentativas: int, tolerancia: float, rng,
           evitar=(), centro=(0.0, 0.0), f0: float = 1.0,
           separacao: float = SEPARACAO):
    """RANSAC de um ponto de fuga: par de retas propoe, o resto vota.

    O voto e ponderado pelo comprimento porque reta longa define direcao muito
    melhor que reta curta, e a cabine tem muito segmento curto de textura.

    `evitar` sao vps ja aceitos: candidato perto de um deles e descartado antes
    de contar voto. f0 e so uma escala para comparar direcoes, nao precisa ser
    a focal verdadeira.
    """
    if len(segs) < 2:
        return None, None
    pesos = np.linalg.norm(segs[:, 2:] - segs[:, :2], axis=1)
    homog = np.hstack([segs[:, :2], np.ones((len(segs), 1))])
    fim = np.hstack([segs[:, 2:], np.ones((len(segs), 1))])
    retas = np.cross(homog, fim)
    proibidas = [_direcao(v, centro, f0) for v in evitar]

    melhor, nota = None, 0.0
    for _ in range(tentativas):
        i, j = rng.choice(len(segs), 2, replace=False)
        vp = np.cross(retas[i], retas[j])
        if np.linalg.norm(vp) < 1e-9:
            continue
        if proibidas:
            atual_dir = _direcao(vp, centro, f0)
            # abs: a direcao oposta da o mesmo ponto de fuga
            if max(abs(float(atual_dir @ d)) for d in proibidas) > np.cos(separacao):
                continue
        dentro = _erro(segs, vp) < tolerancia
        atual = pesos[dentro].sum()
        if atual > nota:
            melhor, nota = vp, atual

    if melhor is None:
        return None, None

    # refino: com todos os apoios, o vp e o vetor que minimiza a distancia as
    # retas deles, ou seja o menor vetor singular da pilha de retas
    dentro = _erro(segs, melhor) < tolerancia
    if dentro.sum() >= 2:
        _, _, vt = np.linalg.svd(retas[dentro] / np.linalg.norm(retas[dentro], axis=1, keepdims=True))
        melhor = vt[-1]
    return melhor / np.linalg.norm(melhor), _erro(segs, melhor) < tolerancia


def _direcao(vp: np.ndarray, centro, f: float) -> np.ndarray:
    """Direcao 3D da familia, normalizada. Serve para comparar dois vps.

    Comparar em pixel nao funciona: dois vps podem estar a milhares de px um do
    outro e mesmo assim ser quase a mesma direcao, porque perto do infinito a
    imagem estica. No espaco de direcoes a comparacao e honesta.
    """
    d = np.array([vp[0] - centro[0] * vp[2], vp[1] - centro[1] * vp[2], f * vp[2]])
    return d / max(np.linalg.norm(d), 1e-9)


def pontos_de_fuga(frame: np.ndarray, quantos: int = 3, tentativas: int = 2000,
                   tolerancia: float = TOLERANCIA, semente: int = 0,
                   separacao: float = SEPARACAO, segs: np.ndarray | None = None):
    """Ate `quantos` pontos de fuga dominantes, do mais apoiado ao menos.

    Tirar os segmentos ja explicados nao basta para nao reencontrar a mesma
    familia: medido nesta cabine, as tres primeiras rodadas devolveram
    (2286,19), (1941,165) e (1939,-11) - o mesmo eixo do transportador tres
    vezes, porque a distorcao da lente espalha a familia alem da tolerancia e a
    sobra ainda e a maior coisa da imagem. Por isso a rodada seguinte tem que
    apontar para uma DIRECAO nova, nao so para segmentos novos.

    `segs` pronto permite juntar as retas de varios frames antes de decidir. A
    camera e fixa e a cabine nao anda, entao a estrutura esta no mesmo lugar em
    todos eles; o que muda de frame para frame e a peca pendurada, que nao tem
    voto suficiente para formar familia sozinha. Um frame so nao basta: medido,
    a familia do transportador tem 85 retas e as outras duas 15 e 16, e com
    esse desequilibrio a focal saiu entre 571 e 1167 px.
    """
    if segs is None:
        segs = segmentos(frame)
    alt, larg = frame.shape[:2]
    centro = (larg / 2.0, alt / 2.0)
    rng = np.random.default_rng(semente)
    achados = []
    restantes = segs
    for _ in range(quantos):
        vp, dentro = _um_vp(restantes, tentativas, tolerancia, rng,
                            [a["vp"] for a in achados], centro, larg, separacao)
        if vp is None or dentro.sum() < MIN_APOIOS:
            break
        achados.append({"vp": vp, "segmentos": restantes[dentro],
                        "apoios": int(dentro.sum())})
        restantes = restantes[~dentro]
    return achados, segs


def focal(vp_a: np.ndarray, vp_b: np.ndarray, centro) -> float | None:
    """Distancia focal em px a partir de dois pontos de fuga ortogonais.

    Vem de uma identidade so: se as duas direcoes sao perpendiculares no mundo,
    os raios que saem do centro optico ate os dois pontos de fuga tambem sao,
    e o produto escalar deles zera. Escrevendo isso com o ponto principal em
    `centro` sobra (a-c).(b-c) + f^2 = 0.

    Devolve None quando o produto e positivo: as duas familias escolhidas nao
    eram ortogonais, ou uma delas e ponto de fuga no infinito.
    """
    if abs(vp_a[2]) < 1e-9 or abs(vp_b[2]) < 1e-9:
        return None
    a = vp_a[:2] / vp_a[2] - np.asarray(centro, float)
    b = vp_b[:2] / vp_b[2] - np.asarray(centro, float)
    produto = float(a @ b)
    return float(np.sqrt(-produto)) if produto < 0 else None


def orientacao(vp_a: np.ndarray, vp_b: np.ndarray, f: float, centro) -> np.ndarray:
    """Rotacao camera->mundo a partir de dois eixos; o terceiro sai do produto
    vetorial.

    A direcao 3D de uma familia paralela e K^-1 vp normalizado. Usar a terceira
    familia medida como terceira coluna parece mais honesto mas nao e: se ela
    for espuria a matriz sai sem ser rotacao. Medido aqui - a terceira familia
    tinha 116 retas contra 457 e 472 das outras duas e apontava para o infinito
    a 45 graus, e a matriz montada com ela nao era ortonormal. Com o produto
    vetorial o resultado e uma rotacao por construcao.

    Gram-Schmidt no segundo eixo porque as duas medidas nunca sao exatamente
    perpendiculares e sem isso a matriz sai levemente torta.
    """
    u = _direcao(vp_a, centro, f)
    v = _direcao(vp_b, centro, f)
    v = v - (u @ v) * u
    v /= max(np.linalg.norm(v), 1e-9)
    R = np.column_stack([u, v, np.cross(u, v)])
    return R


def calibrar(frame: np.ndarray, **kwargs) -> dict:
    """Junta tudo: pontos de fuga -> focal -> abertura -> orientacao."""
    achados, todos = pontos_de_fuga(frame, **kwargs)
    alt, larg = frame.shape[:2]
    centro = (larg / 2.0, alt / 2.0)   # sem tabua nao da para tirar o ponto principal

    vps = [a["vp"] for a in achados]
    apoios = [a["apoios"] for a in achados]
    pares = []
    for i in range(len(vps)):
        for j in range(i + 1, len(vps)):
            valor = focal(vps[i], vps[j], centro)
            # camera de vigilancia fica entre uns 40 e 120 graus de abertura, o
            # que em 1920 px da f entre ~830 e ~2600; a folga aqui e generosa,
            # so serve para cortar par claramente nao ortogonal
            if valor and 0.3 * larg < valor < 3 * larg:
                pares.append({"focal": valor, "familias": (i, j),
                              "apoios": apoios[i] + apoios[j]})

    # o par mais apoiado ganha, nao a mediana: medido nesta cabine, um par bom
    # (929 retas, f=1012 px) e um par espurio (573 retas, f=12760 px) davam
    # mediana 6886 px, ou seja um numero que nenhuma das duas medidas defendia
    melhor = max(pares, key=lambda p: p["apoios"], default=None)
    f = melhor["focal"] if melhor else None

    return {
        "familias": achados,
        "segmentos": todos,
        "centro": centro,
        "focal": f,
        "par": melhor["familias"] if melhor else None,
        "pares": pares,
        "fov_h": float(np.degrees(2 * np.arctan(larg / (2 * f)))) if f else None,
        "R": (orientacao(vps[melhor["familias"][0]], vps[melhor["familias"][1]],
                         f, centro) if melhor else None),
    }


CORES = [(0, 220, 255), (0, 255, 120), (255, 120, 0)]


def desenhar(frame: np.ndarray, saida: dict) -> np.ndarray:
    """Uma familia por cor, cada reta estendida ate o seu ponto de fuga.

    Existe porque numero de focal nao diz se as familias fazem sentido: se o
    detector agrupou a grade do piso com o trilho, so o desenho mostra.
    """
    vista = frame.copy()
    for cor, fam in zip(CORES, saida["familias"]):
        vp = fam["vp"]
        finito = abs(vp[2]) > 1e-9
        alvo = (vp[:2] / vp[2]) if finito else None
        for x1, y1, x2, y2 in fam["segmentos"]:
            cv2.line(vista, (int(x1), int(y1)), (int(x2), int(y2)), cor, 2, cv2.LINE_AA)
            if finito:
                meio = ((x1 + x2) / 2, (y1 + y2) / 2)
                cv2.line(vista, (int(meio[0]), int(meio[1])),
                         (int(alvo[0]), int(alvo[1])), cor, 1, cv2.LINE_AA)
        if finito:
            cv2.circle(vista, (int(alvo[0]), int(alvo[1])), 14, cor, 3)

    linhas = [f"segmentos {len(saida['segmentos'])}   familias {len(saida['familias'])}"]
    for i, fam in enumerate(saida["familias"]):
        vp = fam["vp"]
        onde = (f"({vp[0]/vp[2]:.0f}, {vp[1]/vp[2]:.0f})" if abs(vp[2]) > 1e-9
                else "no infinito")
        linhas.append(f"  familia {i}: {fam['apoios']} retas  vp {onde}")
    if saida["focal"]:
        linhas.append(f"focal {saida['focal']:.0f} px   FOV horizontal {saida['fov_h']:.1f} graus"
                      f"   (familias {saida['par'][0]}+{saida['par'][1]})")
    else:
        linhas.append("focal indeterminada: nenhum par ortogonal")

    fundo = np.zeros((len(linhas) * 30 + 14, 760, 3), np.uint8)
    cv2.addWeighted(fundo, 0.55, vista[:fundo.shape[0], :760], 0.45, 0,
                    vista[:fundo.shape[0], :760])
    for i, texto in enumerate(linhas):
        cv2.putText(vista, texto, (12, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vista


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagem", help="um frame especifico")
    p.add_argument("--dataset", default="dataset_labeled")
    p.add_argument("--camera", default="cabine")
    p.add_argument("--frames", type=int, default=20,
                   help="quantos frames do dataset usar")
    p.add_argument("--por-frame", action="store_true",
                   help="mede cada frame isolado em vez de juntar as retas")
    p.add_argument("--saida", default="capturas/fuga_{camera}.jpg")
    args = p.parse_args()

    if args.imagem:
        caminhos = [args.imagem]
    else:
        caminhos = sorted(glob.glob(os.path.join(args.dataset, f"*_{args.camera}.jpg")))[:args.frames]
    if not caminhos:
        raise SystemExit("Nenhum frame para medir")

    destino = args.saida.format(camera=args.camera)
    os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)

    if args.por_frame:
        focais = []
        primeiro = None
        for caminho in caminhos:
            frame = cv2.imread(caminho)
            if frame is None:
                continue
            atual = calibrar(frame)
            primeiro = primeiro or (atual, frame)
            marca = (f"{atual['focal']:.0f} px  FOV {atual['fov_h']:.1f}"
                     if atual["focal"] else "sem focal")
            print(f"{os.path.basename(caminho)}  familias {len(atual['familias'])}  {marca}")
            if atual["focal"]:
                focais.append(atual["focal"])
        if focais:
            print(f"\nfocal mediana {np.median(focais):.0f} px sobre "
                  f"{len(focais)} frame(s), desvio {np.std(focais):.0f}")
        if primeiro:
            cv2.imwrite(destino, desenhar(primeiro[1], primeiro[0]))
            print(f"desenho em {destino}")
        return

    pilha = []
    frame = None
    for caminho in caminhos:
        atual = cv2.imread(caminho)
        if atual is None:
            continue
        frame = frame if frame is not None else atual
        pilha.append(segmentos(atual))
    if not pilha:
        raise SystemExit("Nenhum frame legivel")

    todas = np.vstack(pilha)
    print(f"{len(pilha)} frames  {len(todas)} retas acumuladas")
    saida = calibrar(frame, segs=todas)

    for i, fam in enumerate(saida["familias"]):
        vp = fam["vp"]
        onde = (f"({vp[0]/vp[2]:8.0f}, {vp[1]/vp[2]:8.0f})" if abs(vp[2]) > 1e-9
                else "no infinito")
        print(f"  familia {i}: {fam['apoios']:4d} retas  vp {onde}")
    for par in saida["pares"]:
        marca = "  <- usado" if par["familias"] == saida["par"] else ""
        print(f"  familias {par['familias'][0]}+{par['familias'][1]}: "
              f"focal {par['focal']:6.0f} px  apoio {par['apoios']}{marca}")

    if saida["focal"]:
        larg = frame.shape[1]
        print(f"\nfocal {saida['focal']:.0f} px   "
              f"FOV horizontal {np.degrees(2 * np.arctan(larg / (2 * saida['focal']))):.1f} graus")
        print(f"para uma peca de 600 px: modelo3d distancia = "
              f"{2 * saida['focal'] / 600:.1f} raios")
        if saida["R"] is not None:
            print("rotacao camera->cabine:")
            for linha in saida["R"]:
                print("   " + "  ".join(f"{v:+.3f}" for v in linha))
    else:
        print("\nNenhum par ortogonal: as familias encontradas nao sao "
              "perpendiculares no mundo")

    cv2.imwrite(destino, desenhar(frame, saida))
    print(f"desenho em {destino}")


if __name__ == "__main__":
    main()
