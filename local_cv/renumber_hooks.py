"""
Inverte a numeracao dos ganchos de uma calibracao.

O `select_hook_points.py` numera os pontos na ordem dos cliques. Se a marcacao
foi feita da esquerda para a direita mas a planta numera os ganchos ao
contrario, todos os ids saem trocados: o que foi marcado como 1 e na verdade o
11, o 2 e o 10, e assim por diante. O sintoma e a deteccao discordar da API de
um jeito que nao melhora por mais que se ajuste limiar.

Este script corrige o arquivo no lugar (id -> N+1-id), guardando um .bak antes.

Uso:
    python local_cv/renumber_hooks.py local_cv/hooks_cabine_11.json
    python local_cv/renumber_hooks.py local_cv/hooks_cabine_*.json --conferir
"""
import argparse
import glob
import json
import shutil


def inverter(caminho: str, conferir: bool) -> None:
    with open(caminho, "r", encoding="utf-8") as f:
        dados = json.load(f)

    hooks = dados["hooks"]
    n = len(hooks)
    invertidos = sorted(({**h, "id": n + 1 - h["id"]} for h in hooks),
                         key=lambda h: h["id"])

    print(f"\n{caminho} ({n} ganchos)")
    for antigo, novo in zip(sorted(hooks, key=lambda h: h["id"]),
                             sorted(invertidos, key=lambda h: h["x"])):
        print(f"  id {antigo['id']:>2} (x={antigo['x']:>4}) -> id {novo['id']:>2}")

    if conferir:
        print("  (--conferir: nada foi gravado)")
        return

    shutil.copy2(caminho, caminho + ".bak")
    dados["hooks"] = invertidos
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2)
    print(f"  gravado. copia do original em {caminho}.bak")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverte a numeracao dos ganchos")
    parser.add_argument("arquivos", nargs="+", help="Calibracoes a corrigir")
    parser.add_argument("--conferir", action="store_true",
                         help="So mostra o que mudaria, sem gravar")
    args = parser.parse_args()

    caminhos = [c for padrao in args.arquivos for c in glob.glob(padrao)]
    if not caminhos:
        raise SystemExit("Nenhum arquivo encontrado.")

    for caminho in caminhos:
        inverter(caminho, args.conferir)


if __name__ == "__main__":
    main()
