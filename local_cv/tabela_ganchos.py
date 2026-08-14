"""Consulta a tabela historica de faixas de gancho por peca (tabela_pecas_ganchos.csv).

Fonte: planilha de programas de robo fornecida pelo usuario (2026-08-10), extraida do
mesmo sistema que atribui PART_NUMBER a gancho. ~155 part numbers, ~310 (part_number,
faixa) distintos. Position_name "Prog." = posicao programada (a maioria); "Pos A"/"Pos B"
aparecem em poucos part numbers como sub-posicoes dentro da faixa "Prog." mais ampla.

Uso pretendido: VERIFICACAO barata. A API ja diz o part_number esperado em cada evento;
aqui a gente checa se o gancho que ELA PROPRIA relata bate com o que esse part_number
historicamente ocupou. Fora do historico = alerta (dado estranho ou robo reprogramado),
nao prova de peca errada sozinho.

CUIDADO (visto nos dados): a mesma peca pode ter faixas MUITO diferentes conforme o
carro (8 vs 11 ganchos). Ex.: FRAME AS-MTG aparece em faixas curtas [2,3][4,5][7,8][10,11]
(part 3072881) E em faixas longas [5,11]/[6,11] (part 3519810, 3936315, 4488655...).
NAO e erro de leitura - carros de capacidades diferentes usam o mesmo nome de peca com
programacao de gancho diferente. Portanto comparar por PART_NUMBER exato (nao por nome)
sempre que possivel; usar nome so como fallback mais fraco.
"""
import csv
import os

CAMINHO_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabela_pecas_ganchos.csv")


def carregar(caminho: str = CAMINHO_PADRAO) -> list[dict]:
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for l in linhas:
        l["hook_start"] = int(l["hook_start"])
        l["hook_end"] = int(l["hook_end"])
        l["part_number"] = l["part_number"].strip()
        l["part_name"] = l["part_name"].strip()
    return linhas


def faixas_do_part_number(part_number: str, linhas: list[dict] | None = None) -> set[tuple[int, int]]:
    linhas = linhas if linhas is not None else carregar()
    part_number = str(part_number).strip()
    return {(l["hook_start"], l["hook_end"]) for l in linhas if l["part_number"] == part_number}


def faixas_do_nome(part_name: str, linhas: list[dict] | None = None) -> set[tuple[int, int]]:
    linhas = linhas if linhas is not None else carregar()
    return {(l["hook_start"], l["hook_end"]) for l in linhas if l["part_name"] == part_name}


def nomes_por_faixa(hook_start: int, hook_end: int, linhas: list[dict] | None = None) -> set[str]:
    """Que PART_NAME ja usaram exatamente essa faixa historicamente (match exato)."""
    linhas = linhas if linhas is not None else carregar()
    return {l["part_name"] for l in linhas if l["hook_start"] == hook_start and l["hook_end"] == hook_end}


def checar_vao(part_number: str, ganchos_detectados: list[int], linhas: list[dict] | None = None) -> dict:
    """Compara a faixa OCUPADA DETECTADA (min..max dos ganchos vistos como ocupados)
    contra o historico desse part_number. Nao decide sozinho "peca errada" - devolve
    o material pra quem for julgar (monitor/alerta).
    """
    linhas = linhas if linhas is not None else carregar()
    conhecidas = faixas_do_part_number(part_number, linhas)
    if not ganchos_detectados:
        return {"esperado": None, "motivo": "sem ganchos detectados", "faixas_conhecidas": conhecidas}
    span = (min(ganchos_detectados), max(ganchos_detectados))
    if not conhecidas:
        return {"esperado": None, "motivo": "part_number sem historico na tabela",
                "faixa_detectada": span, "faixas_conhecidas": conhecidas}
    bate_exato = span in conhecidas
    # tolerancia: aceita se o span detectado esta CONTIDO em alguma faixa conhecida
    # (deteccao local pode nao pegar as pontas exatas do vao)
    contido = any(span[0] >= c[0] and span[1] <= c[1] for c in conhecidas)
    return {
        "esperado": bate_exato or contido,
        "exato": bate_exato,
        "faixa_detectada": span,
        "faixas_conhecidas": conhecidas,
    }


if __name__ == "__main__":
    linhas = carregar()
    print(f"{len(linhas)} linhas, {len({l['part_number'] for l in linhas})} part numbers distintos")
    print("exemplo AXLE GP-FRONT:", sorted(faixas_do_nome("AXLE GP-FRONT", linhas)))
    print("exemplo FRAME AS-MTG:", sorted(faixas_do_nome("FRAME AS-MTG", linhas)))
