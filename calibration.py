"""Calibracao dos ganchos: carregamento, escolha automatica e resolucao do arquivo.

Os carros da linha tem quantidades diferentes de ganchos (8 ou 11), e a
geometria muda junto: um carro de 8 ganchos nao tem os ganchos nas mesmas
posicoes em pixel que os 8 primeiros de um carro de 11. Por isso a calibracao
e por (camera, quantidade de ganchos do carro):

    local_cv/hooks_cabine_8.json
    local_cv/hooks_cabine_11.json

Como nao da para saber pela API qual carro esta na linha, a escolha e feita
pela imagem: testa todas as calibracoes e fica com a que tem maior confianca,
isto e, aquela cujos pontos realmente caem sobre estrutura de gancho em vez de
parede lisa.
"""
import json
import os

import cv2
import numpy as np

CALIB_DIR = "local_cv"


def hooks_path_for(camera: str, car_hooks: int | None = None) -> str:
    """Caminho da calibracao. Sem car_hooks, cai no arquivo sem sufixo."""
    if car_hooks is None:
        return os.path.join(CALIB_DIR, f"hooks_{camera}.json")
    return os.path.join(CALIB_DIR, f"hooks_{camera}_{car_hooks}.json")


def available_calibrations(camera: str) -> dict[int, str]:
    """Calibracoes existentes para a camera, indexadas pela quantidade de ganchos."""
    found = {}
    if not os.path.isdir(CALIB_DIR):
        return found

    prefix = f"hooks_{camera}_"
    for name in os.listdir(CALIB_DIR):
        if not (name.startswith(prefix) and name.endswith(".json")):
            continue
        sufixo = name[len(prefix):-len(".json")]
        if sufixo.isdigit():
            found[int(sufixo)] = os.path.join(CALIB_DIR, name)
    return found


def load_hooks(hooks_path: str) -> list[dict]:
    with open(hooks_path, "r", encoding="utf-8") as f:
        return json.load(f)["hooks"]


def thresholds_path_for(hooks_path: str) -> str:
    """Arquivo de limiares aprendidos que acompanha uma calibracao.

    local_cv/hooks_cabine_11.json  ->  local_cv/limiares_cabine_11.json
    """
    pasta, nome = os.path.split(hooks_path)
    return os.path.join(pasta, nome.replace("hooks_", "limiares_", 1))


def load_thresholds(hooks_path: str) -> dict[int, float]:
    """Limiares por gancho aprendidos dos dados, se ja existirem.

    Vazio quando ainda nao houve aprendizado - nesse caso quem chama usa o
    limiar geral. So aparecem aqui os ganchos em que o score de fato separa
    ocupado de vazio; os demais ficam de fora de proposito (ver
    learn_thresholds.py).
    """
    caminho = thresholds_path_for(hooks_path)
    if not os.path.exists(caminho):
        return {}

    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return {int(k): float(v) for k, v in dados.get("limiares", {}).items()}


def candidate_paths(camera: str, explicit: str | None = None,
                     car_hooks: int | None = None) -> list[str]:
    """Calibracoes que devem ser consideradas para esta camera."""
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"Calibracao nao encontrada: {explicit}")
        return [explicit]

    if car_hooks is not None:
        caminho = hooks_path_for(camera, car_hooks)
        if not os.path.exists(caminho):
            raise SystemExit(
                f"Nao existe calibracao de {car_hooks} ganchos para '{camera}' ({caminho}).\n"
                f"Calibracoes disponiveis: {sorted(available_calibrations(camera)) or 'nenhuma'}\n"
                f"Crie com: python local_cv/select_hook_points.py "
                f"--source <rtsp> --out {caminho}")
        return [caminho]

    disponiveis = available_calibrations(camera)
    if disponiveis:
        return [disponiveis[n] for n in sorted(disponiveis)]

    legado = hooks_path_for(camera)
    if os.path.exists(legado):
        return [legado]

    raise SystemExit(
        f"Nenhuma calibracao encontrada para '{camera}'.\n"
        f"Crie com: python local_cv/select_hook_points.py "
        f"--source <rtsp> --out {hooks_path_for(camera, 8)}")


def hook_confidence(frame, hooks: list[dict], radius: int = 14) -> float:
    """Media de estrutura encontrada nos pontos calibrados.

    Um ponto que cai sobre um gancho pega metal contra a parede e gera muitas
    bordas; um ponto que cai sobre parede lisa gera quase nenhuma. Serve para
    decidir qual calibracao corresponde ao carro que esta na linha agora.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores = []
    for hook in hooks:
        y1, y2 = max(0, hook["y"] - radius), min(gray.shape[0], hook["y"] + radius)
        x1, x2 = max(0, hook["x"] - radius), min(gray.shape[1], hook["x"] + radius)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            scores.append(0.0)
            continue
        edges = cv2.Canny(cv2.GaussianBlur(patch, (5, 5), 0), 50, 150)
        scores.append(float(np.count_nonzero(edges)) / edges.size * 100.0)

    return float(np.mean(scores)) if scores else 0.0


def pick_calibration(frame, camera: str, explicit: str | None = None,
                      car_hooks: int | None = None, verbose: bool = True):
    """Escolhe a calibracao que melhor casa com o carro presente no frame.

    Retorna (caminho, hooks, confianca).
    """
    caminhos = candidate_paths(camera, explicit, car_hooks)

    avaliadas = []
    for caminho in caminhos:
        hooks = load_hooks(caminho)
        avaliadas.append((hook_confidence(frame, hooks), caminho, hooks))
    avaliadas.sort(key=lambda item: item[0], reverse=True)

    confianca, caminho, hooks = avaliadas[0]

    if verbose and len(avaliadas) > 1:
        for c, p, h in avaliadas:
            marca = "<--" if p == caminho else "   "
            print(f"  {marca} {os.path.basename(p)}: {len(h)} ganchos, confianca {c:.1f}")

        segunda = avaliadas[1][0]
        if segunda > 0 and (confianca - segunda) / confianca < 0.2:
            print("  AVISO: as calibracoes ficaram proximas, a escolha e incerta.")

    if verbose:
        print(f"calibracao: {caminho} ({len(hooks)} ganchos, confianca {confianca:.1f})")

    return caminho, hooks, confianca


class EncaixeDaCalibracao:
    """Quao bem a calibracao esta encaixando AGORA, comparado ao seu proprio pico.

    Um limiar absoluto nessa medida nao funciona. Ela e densidade de borda nos
    pontos calibrados, e o valor tipico muda com o carro, a luz e a fumaca:
    medido 14-16 num dia e 10.5 em outro, com o carro igualmente parado. Um
    corte fixo em 10 marcou 16% dos frames como "carro em movimento" com o
    carro parado.

    Aqui a referencia e o pico recente da propria calibracao, que decai com o
    tempo. Carro bem posicionado empurra o pico para cima; carro entrando ou
    saindo derruba a confianca para bem abaixo dele, que e o sinal de verdade.
    O decaimento evita que um pico antigo de um carro que ja foi embora fique
    valendo para sempre.
    """

    def __init__(self, meia_vida: float = 300.0, fracao: float = 0.7):
        self.meia_vida = meia_vida
        self.fracao = fracao
        self.pico = 0.0
        self._ultimo = None

    def avaliar(self, confianca: float, agora: float | None = None) -> float:
        """Atualiza o pico e devolve o encaixe atual, de 0 a 1."""
        import time as _time

        agora = _time.monotonic() if agora is None else agora
        if self._ultimo is not None and self.pico > 0:
            self.pico *= 0.5 ** ((agora - self._ultimo) / self.meia_vida)
        self._ultimo = agora

        self.pico = max(self.pico, confianca)
        return min(1.0, confianca / self.pico) if self.pico > 0 else 0.0

    def confiavel(self, encaixe: float) -> bool:
        return encaixe >= self.fracao


class SeletorDeCalibracao:
    """Escolhe a calibracao com memoria, para nao trocar a cada frame.

    pick_calibration sozinha pega o maximo de cada frame, e como a confianca
    oscila (operador passando na frente, fumaca, carro chegando), ela fica
    alternando entre a calibracao de 8 e a de 11 de um frame para o outro. Isso
    estraga tudo que depende da geometria: os circulos pulam e qualquer recorte
    de peca cai no chao.

    A calibracao atual so e trocada quando outra ganha por uma margem folgada,
    em varios frames seguidos, E encaixando bem pelo proprio padrao dela. Essa
    ultima condicao e a que importa: quando o carro esta entrando ou saindo,
    nenhuma calibracao casa e todas caem juntas; aceitar a vencedora nessa hora
    e trocar por causa de ruido, nao por causa de carro novo.
    """

    def __init__(self, camera: str, explicit: str | None = None,
                 car_hooks: int | None = None, margem: float = 1.3,
                 confirmacoes: int = 4, fracao: float = 0.7):
        self.caminhos = candidate_paths(camera, explicit, car_hooks)
        self.margem = margem
        self.confirmacoes = confirmacoes
        self.encaixes = {c: EncaixeDaCalibracao(fracao=fracao) for c in self.caminhos}
        self.atual: str | None = None
        self._candidato: str | None = None
        self._seguidas = 0

    def escolher(self, frame) -> tuple[str, list[dict], float, float]:
        """Retorna (caminho, hooks, confianca, encaixe) da calibracao vigente."""
        medidas = {}
        for caminho in self.caminhos:
            hooks = load_hooks(caminho)
            confianca = hook_confidence(frame, hooks)
            medidas[caminho] = (confianca, hooks,
                                self.encaixes[caminho].avaliar(confianca))

        melhor = max(medidas, key=lambda c: medidas[c][0])

        if self.atual is None or len(self.caminhos) == 1:
            self.atual = melhor
        elif melhor != self.atual:
            supera = (medidas[melhor][0] >= medidas[self.atual][0] * self.margem
                      and self.encaixes[melhor].confiavel(medidas[melhor][2]))
            if supera and melhor == self._candidato:
                self._seguidas += 1
            elif supera:
                self._candidato, self._seguidas = melhor, 1
            else:
                self._candidato, self._seguidas = None, 0

            if self._seguidas >= self.confirmacoes:
                self.atual = melhor
                self._candidato, self._seguidas = None, 0
        else:
            self._candidato, self._seguidas = None, 0

        confianca, hooks, encaixe = medidas[self.atual]
        return self.atual, hooks, confianca, encaixe

    def confiavel(self, encaixe: float) -> bool:
        return self.encaixes[self.atual].confiavel(encaixe)

    def resetar(self) -> None:
        """Esquece a escolha - use ao trocar de camera."""
        self.atual = None
        self._candidato, self._seguidas = None, 0


