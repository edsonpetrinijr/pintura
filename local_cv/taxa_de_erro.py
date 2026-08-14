"""Mede com que frequencia a carga real diverge do que a API diz.

Este e o entregavel do modo silencioso: nao alarma ninguem, so conta. Responde
"quantas vezes a carga real nao bate com a API", que e o numero que decide se o
error proofing vale a pena.

O QUE ESTE SCRIPT NAO PODE FAZER SOZINHO
----------------------------------------
Uma divergencia e a soma de duas coisas diferentes:

    divergencia = erro do operador + erro do nosso detector

O CSV nao sabe qual das duas aconteceu - as duas produzem exatamente a mesma
linha. So um humano olhando a imagem separa. Por isso o script termina gerando
uma FILA DE ADJUDICACAO: um CSV com uma coluna `veredicto` em branco para
alguem preencher com OPERADOR ou DETECTOR. Sem isso a taxa medida e um teto,
nao a taxa de erro do operador.

UNIDADE DE MEDIDA
-----------------
Um carro parado dez minutos aparece dez vezes no CSV. Contar linhas inflaria a
taxa em uma ordem de grandeza. A unidade aqui e o EPISODIO: sequencia de
capturas seguidas com o mesmo carro e o mesmo estado da API.

Uso:
    .venv\\Scripts\\python.exe local_cv\\taxa_de_erro.py
    .venv\\Scripts\\python.exe local_cv\\taxa_de_erro.py --fila logs/adjudicar.csv
"""
import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from local_cv.stability import BAIXA

FILA_CAMPOS = ["episodio", "inicio", "camera", "number_car", "tipo", "ganchos",
               "api_ganchos_ocupados", "api_pecas", "imagem", "veredicto"]


def ler(caminho: str) -> list[dict]:
    if not os.path.exists(caminho):
        raise SystemExit(f"Nao achei {caminho}. Rode local_cv/collect_validation.py antes.")
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def episodios(linhas: list[dict]):
    """Agrupa capturas seguidas do mesmo carro e mesmo estado da API."""
    atual, chave_atual = [], None
    for linha in linhas:
        chave = (linha.get("camera", ""), linha.get("number_car", ""),
                 linha.get("api_ganchos_ocupados", ""))
        if chave != chave_atual and atual:
            yield chave_atual, atual
            atual = []
        chave_atual = chave
        atual.append(linha)
    if atual:
        yield chave_atual, atual


def avaliar(bloco: list[dict]) -> dict | None:
    """Compara a ultima leitura utilizavel do episodio com a API.

    Devolve None quando o episodio nao vale como medida: API fora do ar ou
    leitura sem confianca. Contar esses como "sem divergencia" mentiria para o
    lado otimista, que e justamente o lado perigoso.
    """
    usaveis = [l for l in bloco if l.get("api_ok") == "1" and l.get("certeza") != BAIXA]
    if not usaveis:
        return None

    # A ultima captura do episodio e a mais estavel: o carro ja parou.
    carimbo = usaveis[-1]["timestamp"]
    ultimas = [l for l in usaveis if l["timestamp"] == carimbo]

    falta, sobra = [], []
    for l in ultimas:
        local = l.get("ocupado_local") == "1"
        api = l.get("ocupado_api") == "1"
        if api and not local:
            falta.append(l["gancho"])
        elif local and not api:
            sobra.append(l["gancho"])

    imagem = next((l["imagem"] for l in reversed(usaveis) if l.get("imagem")), "")
    return {
        "inicio": bloco[0]["timestamp"],
        "camera": ultimas[0].get("camera", ""),
        "number_car": ultimas[0].get("number_car", ""),
        "api_ganchos_ocupados": ultimas[0].get("api_ganchos_ocupados", ""),
        "api_pecas": ultimas[0].get("api_pecas", ""),
        "falta": falta,
        "sobra": sobra,
        "imagem": imagem,
        "capturas": len(bloco),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="logs/validacao_ganchos.csv")
    p.add_argument("--fila", default="logs/adjudicar.csv",
                   help="Onde gravar a fila de adjudicacao (o que um humano precisa olhar)")
    args = p.parse_args()

    linhas = ler(args.csv)
    if not linhas:
        raise SystemExit("CSV vazio.")
    if "number_car" not in linhas[0]:
        print("AVISO: CSV sem a coluna number_car (formato antigo). Agrupando so "
              "pelo estado da API, o que junta carros diferentes com a mesma carga.\n")

    medidos, descartados = [], 0
    for _, bloco in episodios(linhas):
        r = avaliar(bloco)
        if r is None:
            descartados += 1
        else:
            medidos.append(r)

    if not medidos:
        raise SystemExit(f"Nenhum episodio utilizavel ({descartados} descartados).")

    divergentes = [r for r in medidos if r["falta"] or r["sobra"]]
    so_falta = [r for r in divergentes if r["falta"] and not r["sobra"]]
    so_sobra = [r for r in divergentes if r["sobra"] and not r["falta"]]
    ambos = [r for r in divergentes if r["falta"] and r["sobra"]]

    n = len(medidos)
    print(f"episodios utilizaveis : {n}")
    print(f"descartados           : {descartados}  (API fora do ar ou leitura sem confianca)")
    print(f"periodo               : {medidos[0]['inicio']}  ate  {medidos[-1]['inicio']}\n")

    print(f"com divergencia       : {len(divergentes)}/{n} ({100*len(divergentes)/n:.1f}%)")
    print(f"  API diz cheio, vimos vazio : {len(so_falta):4d}  <- peca faltando OU deteccao falhou")
    print(f"  vimos cheio, API calada    : {len(so_sobra):4d}  <- obstaculo nao previsto OU falso positivo")
    print(f"  os dois no mesmo carro     : {len(ambos):4d}  <- cheira a peca no gancho errado\n")

    if divergentes:
        quem = Counter(g for r in divergentes for g in r["falta"] + r["sobra"])
        print("ganchos que mais divergem (se concentrar em poucos, e calibracao, nao operador):")
        for gancho, vezes in quem.most_common(8):
            print(f"  gancho {gancho:>3s}: {vezes}x")
        print()

    os.makedirs(os.path.dirname(args.fila) or ".", exist_ok=True)
    with open(args.fila, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FILA_CAMPOS)
        w.writeheader()
        for i, r in enumerate(divergentes, 1):
            tipo = ("falta+sobra" if r["falta"] and r["sobra"]
                    else "falta" if r["falta"] else "sobra")
            w.writerow({
                "episodio": i,
                "inicio": r["inicio"],
                "camera": r["camera"],
                "number_car": r["number_car"],
                "tipo": tipo,
                "ganchos": ";".join(r["falta"] + r["sobra"]),
                "api_ganchos_ocupados": r["api_ganchos_ocupados"],
                "api_pecas": r["api_pecas"],
                "imagem": r["imagem"],
                "veredicto": "",
            })

    sem_imagem = sum(1 for r in divergentes if not r["imagem"])
    print(f"fila de adjudicacao: {args.fila} ({len(divergentes)} casos)")
    if sem_imagem:
        print(f"  ATENCAO: {sem_imagem} sem imagem -> nao da para adjudicar. "
              f"Rode o coletor sem --somente-mudancas.")
    print("\nPreencha a coluna `veredicto` com OPERADOR ou DETECTOR olhando a imagem.")
    print("So depois disso o numero acima vira taxa de erro do operador; hoje ele")
    print("e um TETO, porque inclui os nossos proprios erros.")


if __name__ == "__main__":
    main()
