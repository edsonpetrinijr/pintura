"""Cruza a leitura de duas cameras que olham a mesma fila de ganchos.

A ideia: cada camera enxerga bem o que esta perto dela e mal o que esta longe.
Na .45 os ganchos do fundo ficam espremidos em poucos pixels e viram chute; na
.46, que olha a mesma fila pela diagonal, esses mesmos ganchos aparecem maiores.
Cruzando as duas, cada gancho e decidido principalmente pela camera que o ve
melhor.

O peso NAO e ajustado na mao. Ele sai da propria calibracao: o espacamento em
pixel entre ganchos vizinhos e uma medida direta de quao perto aquele trecho da
fila esta da camera. Gancho com 200 px de folga para o vizinho esta perto e
tem area de sobra para medir; gancho com 30 px esta no fundo e qualquer erro de
um pixel ja contamina a leitura.

Alem da geometria, entra a validade do momento: se a calibracao de uma camera
nao esta encaixando agora (carro entrando, porta aberta, fumaca), o peso dela
cai para zero e a outra decide sozinha.

AVISO - FALHA MEDIDA EM 2026-08-05, POR ISSO A FUSAO ESTA DESLIGADA POR PADRAO
NO MONITOR (--fusao para ligar). O paragrafo acima descreve a intencao, nao o
que o codigo faz. O peso vem SO do espacamento dos ganchos; o unico freio e
`EncaixeDaCalibracao`, que normaliza cada vista contra o PROPRIO pico recente.
Uma camera consistentemente ruim, ou cega por neblina de tinta durante um ciclo
inteiro, continua marcando encaixe ~1.0 contra o proprio pico baixo e mantem
poder de voto total. Medido no mesmo instante:

    .45  confianca 15.2  detectou 4/11 ocupados
    .46  confianca  7.4  detectou 1/11 ocupados (cabine em neblina de tinta)

e como nos ganchos 1-4 o peso da .46 e 0.89 contra 0.17 da .45, a vista cega
ANULOU deteccao boa da vista limpa. Consertar exige o peso levar em conta o
quanto a vista esta enxergando agora - e isso precisa de dado da .46, que hoje
nao existe (zero imagens dela no dataset).
"""
import math

from local_cv.stability import ALTA, BAIXA, MEDIA


def pesos_por_distancia(hooks: list[dict]) -> dict[int, float]:
    """Quanto vale a leitura de cada gancho nesta vista, de 0 a 1.

    Usa o espacamento ate os ganchos vizinhos como proxy de distancia: quanto
    mais perto da camera, mais pixels separam um gancho do outro. Normaliza
    pelo maior espacamento da propria vista, entao o resultado e relativo
    aquela camera e pode ser comparado entre cameras diferentes.
    """
    if len(hooks) < 2:
        return {h["id"]: 1.0 for h in hooks}

    ordenados = sorted(hooks, key=lambda h: h["id"])
    pontos = [(h["id"], float(h["x"]), float(h["y"])) for h in ordenados]

    espacos = {}
    for i, (hook_id, x, y) in enumerate(pontos):
        vizinhos = []
        if i > 0:
            vizinhos.append(math.dist((x, y), pontos[i - 1][1:]))
        if i < len(pontos) - 1:
            vizinhos.append(math.dist((x, y), pontos[i + 1][1:]))
        espacos[hook_id] = sum(vizinhos) / len(vizinhos)

    maior = max(espacos.values()) or 1.0
    return {k: v / maior for k, v in espacos.items()}


def margem_relativa(resultado: dict, limiar: float) -> float:
    """Quao decidida esta a leitura: negativo vazio, positivo ocupado.

    Dividido pelo limiar para que cameras com escalas de score diferentes
    possam ser comparadas. Zero significa em cima do limiar, isto e, sem
    opiniao - e uma camera sem opiniao nao deve pesar na decisao.
    """
    if limiar <= 0:
        return 0.0
    return (resultado["score"] - limiar) / limiar


def fundir(vistas: dict[str, dict], encaixe_min: float = 0.7,
            dominancia: float = 2.0, principal: str | None = None) -> list[dict]:
    """Junta as leituras de varias vistas do mesmo carro.

    `vistas` e {nome_da_camera: {"results", "hooks", "limiares", "threshold",
    "confianca"}}, onde "confianca" e o ENCAIXE de 0 a 1 daquela vista (o quanto
    a calibracao esta casando agora comparada ao proprio pico recente). Numero
    absoluto nao serve aqui: a medida bruta muda com carro, luz e fumaca.
    Devolve uma lista no formato dos results, com os campos extras `fontes`,
    `peso` e `divergencia`.

    A DECISAO vem das duas cameras, mas a GEOMETRIA (`point`, `window`) vem
    sempre da vista `principal` - sao coordenadas em pixel, e as da outra
    camera desenhariam no lugar errado sobre a imagem exibida.

    Quando as duas cameras discordam, ganha a de maior peso - mas so vale como
    leitura firme se ela for `dominancia` vezes mais pesada que a outra. Perto
    do empate a divergencia e assumida: melhor dizer que nao sabemos do que
    tirar cara ou coroa.
    """
    principal = principal or next(iter(vistas))
    pesos = {nome: pesos_por_distancia(v["hooks"]) for nome, v in vistas.items()}
    geometria = {r["id"]: r for r in vistas[principal]["results"]}

    por_gancho = {}
    for nome, vista in vistas.items():
        # Encaixe ruim = os pontos calibrados nao estao caindo sobre ganchos.
        # A vista inteira sai da votacao, e nao gancho a gancho.
        if vista["confianca"] < encaixe_min:
            continue
        for r in vista["results"]:
            limiar = vista["limiares"].get(r["id"], vista["threshold"])
            peso = pesos[nome].get(r["id"], 0.0)
            por_gancho.setdefault(r["id"], []).append(
                (nome, peso, margem_relativa(r, limiar), r))

    fundidos = []
    for hook_id in sorted(por_gancho):
        votos = por_gancho[hook_id]
        base = geometria.get(hook_id)
        if base is None:
            # Gancho que so a outra vista tem (calibracao de 8 contra uma de
            # 11). A LEITURA dela continua valendo, mas `point` e `window` sao
            # pixels de outra imagem: copiar aqui desenharia o circulo em cima
            # da imagem errada. Sem geometria propria, sobra so o painel.
            base = {**max(votos, key=lambda v: v[1])[3],
                    "point": None, "window": None}

        uteis = [v for v in votos if v[1] > 0]
        if not uteis:
            fundidos.append({**base, "certeza": BAIXA, "fontes": [],
                              "peso": 0.0, "divergencia": False})
            continue

        total = sum(p for _, p, _, _ in uteis)
        media = sum(p * m for _, p, m, _ in uteis) / total
        ocupado = media > 0

        divergencia = len({m > 0 for _, _, m, _ in uteis}) > 1
        vencedor = max(uteis, key=lambda v: v[1] * abs(v[2]))

        if divergencia:
            # Uma camera diz cheio e a outra diz vazio. So aceito se a que
            # decidiu enxerga esse gancho claramente melhor que a outra.
            perdedor = min(uteis, key=lambda v: v[1])
            certeza = MEDIA if vencedor[1] >= dominancia * max(perdedor[1], 1e-6) else BAIXA
            ocupado = vencedor[2] > 0
        elif len(uteis) > 1:
            # Duas cameras independentes concordando vale mais que uma so.
            certeza = ALTA if all(v[3]["certeza"] != BAIXA for v in uteis) else MEDIA
        else:
            certeza = uteis[0][3]["certeza"]

        fundidos.append({
            **base,
            "occupied": ocupado,
            "certeza": certeza,
            "score": vencedor[3]["score"],
            "fontes": [nome for nome, _, _, _ in uteis],
            "peso": round(total, 3),
            "divergencia": divergencia,
        })

    # Gancho que so existe na outra vista nao pode ser desenhado aqui, mas a
    # leitura dele continua valendo - por isso entra na lista mesmo assim.
    for hook_id, r in geometria.items():
        if hook_id not in por_gancho:
            fundidos.append({**r, "certeza": BAIXA, "fontes": [],
                              "peso": 0.0, "divergencia": False})

    return sorted(fundidos, key=lambda r: r["id"])
