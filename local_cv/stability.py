"""
Estabiliza a leitura de cada gancho no tempo e informa o quanto ela e confiavel.

O problema: um gancho cujo score fica em cima do limiar troca de estado a cada
frame, e a leitura vira ruido. Foi exatamente o caso do gancho 6. Isso NAO se
resolve mexendo no limiar, porque o score dele realmente oscila em volta de
qualquer valor que se escolha ali.

Tres medidas:

  1. decide pela MEDIANA dos ultimos N scores, e nao pelo score instantaneo,
     entao um frame ruim sozinho nao muda a leitura;
  2. histerese: so vira OCUPADO acima de limiar+margem e so vira VAZIO abaixo
     de limiar-margem. Entre os dois mantem o estado anterior, o que impede o
     vai-e-vem de quem esta em cima da linha divisoria;
  3. reporta a certeza da leitura em vez de devolver so um sim/nao. Um gancho
     na zona morta continua sendo mostrado, mas marcado como INCERTO - a
     resposta honesta e "nao sei", nao um chute com cara de resposta.
"""
from collections import defaultdict, deque

import numpy as np

ALTA = "ALTA"
MEDIA = "MEDIA"
BAIXA = "BAIXA"


class HookStabilizer:
    """Guarda o historico recente de cada gancho e devolve a leitura estavel."""

    def __init__(self, janela: int = 9, margem: float = 1.0):
        self.janela = janela
        self.margem = margem
        self._scores: dict[int, deque] = defaultdict(lambda: deque(maxlen=janela))
        self._estado: dict[int, bool] = {}
        self._trocas: dict[int, int] = defaultdict(int)

    def update(self, hook_id: int, score: float, threshold: float) -> dict:
        hist = self._scores[hook_id]
        hist.append(score)

        mediana = float(np.median(hist))
        anterior = self._estado.get(hook_id)
        distancia = mediana - threshold

        if distancia >= self.margem:
            estado = True
        elif distancia <= -self.margem:
            estado = False
        else:
            # Zona morta: sem evidencia suficiente para mudar de ideia.
            estado = anterior if anterior is not None else mediana > threshold

        na_zona_morta = abs(distancia) < self.margem

        votos_ocupado = sum(1 for s in hist if s > threshold)
        concordantes = max(votos_ocupado, len(hist) - votos_ocupado)
        consistencia = concordantes / len(hist)

        if anterior is not None and estado != anterior:
            self._trocas[hook_id] += 1
        self._estado[hook_id] = estado

        folga = abs(distancia) / self.margem if self.margem else 99.0

        if len(hist) < 3:
            certeza = BAIXA
        elif consistencia >= 0.9 and folga >= 1.0:
            certeza = ALTA
        elif consistencia >= 0.7 and not na_zona_morta:
            certeza = MEDIA
        else:
            certeza = BAIXA

        return {
            "occupied": estado,
            "score": mediana,
            "score_bruto": score,
            "certeza": certeza,
            "consistencia": consistencia,
            "concordantes": concordantes,
            "zona_morta": na_zona_morta,
            "trocas": self._trocas[hook_id],
            "amostras": len(hist),
        }

    def aplicar(self, results: list[dict], threshold: float,
                 por_gancho: dict[int, float] | None = None) -> list[dict]:
        """Passa a saida do analyze() pelo estabilizador, preservando o resto.

        `por_gancho` permite um limiar proprio para cada gancho (aprendido de
        dados por learn_thresholds.py); quem nao tiver limiar proprio usa o
        limiar geral.
        """
        estaveis = []
        for r in results:
            limiar = (por_gancho or {}).get(r["id"], threshold)
            info = self.update(r["id"], r["score"], limiar)
            estaveis.append({**r, **info, "threshold": limiar})
        return estaveis
