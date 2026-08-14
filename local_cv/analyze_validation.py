"""
Mede a acuracia da deteccao local contra a API e verifica se ha algum atraso
entre as duas.

A API passou a refletir o conteudo da cabine, entao o esperado e que o melhor
resultado apareca no atraso zero. Se o pico aparecer deslocado, alguma coisa
ainda esta fora de sincronia e vale investigar.

Se o melhor atraso nao for claramente melhor que os vizinhos, o resultado nao
significa nada ainda: colete mais dados.

Uso:
    python local_cv/analyze_validation.py
    python local_cv/analyze_validation.py --max-lag 40 --step 2
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta


def carregar(csv_path: str):
    """Agrupa o CSV por instante de captura."""
    capturas = defaultdict(lambda: {"local": {}, "api": set()})
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = datetime.fromisoformat(row["timestamp"])
            capturas[t]["local"][int(row["gancho"])] = row["ocupado_local"] == "1"
            api = row["api_ganchos_ocupados"]
            capturas[t]["api"] = {int(h) for h in api.split(";") if h}
    return dict(sorted(capturas.items()))


def api_em(capturas: dict, alvo: datetime, tolerancia: timedelta):
    """Estado da API mais proximo do instante alvo, se houver algum perto."""
    melhor, menor = None, None
    for t, dados in capturas.items():
        d = abs(t - alvo)
        if menor is None or d < menor:
            melhor, menor = dados["api"], d
    if menor is None or menor > tolerancia:
        return None
    return melhor


def acuracia_com_atraso(capturas: dict, atraso_min: float, tolerancia: timedelta):
    acertos = comparacoes = 0
    for t, dados in capturas.items():
        api = api_em(capturas, t - timedelta(minutes=atraso_min), tolerancia)
        if api is None:
            continue
        for gancho, ocupado in dados["local"].items():
            acertos += int(ocupado == (gancho in api))
            comparacoes += 1
    return (acertos / comparacoes * 100 if comparacoes else 0.0), comparacoes


def main() -> None:
    parser = argparse.ArgumentParser(description="Descobre o atraso API -> cabine")
    parser.add_argument("--csv", default="logs/validacao_ganchos.csv")
    parser.add_argument("--max-lag", type=float, default=30.0, help="Atraso maximo testado, em minutos")
    parser.add_argument("--step", type=float, default=1.0, help="Passo da varredura, em minutos")
    parser.add_argument("--tolerancia", type=float, default=1.0,
                         help="Quao perto do instante alvo a amostra da API precisa estar, em minutos")
    args = parser.parse_args()

    capturas = carregar(args.csv)
    if len(capturas) < 10:
        raise SystemExit(f"Só {len(capturas)} capturas em {args.csv}. Colete mais antes de analisar.")

    print(f"{len(capturas)} capturas, de {min(capturas)} a {max(capturas)}\n")
    print("%8s %10s %12s" % ("atraso", "acuracia", "comparacoes"))

    tolerancia = timedelta(minutes=args.tolerancia)
    resultados = []
    atraso = 0.0
    while atraso <= args.max_lag:
        acc, n = acuracia_com_atraso(capturas, atraso, tolerancia)
        resultados.append((acc, atraso, n))
        print("%6.1fmin %9.1f%% %12d" % (atraso, acc, n))
        atraso += args.step

    melhor_acc, melhor_atraso, _ = max(resultados)
    media = sum(a for a, _, _ in resultados) / len(resultados)

    print(f"\nMelhor atraso: {melhor_atraso:.1f} min, com {melhor_acc:.1f}% de acerto")
    if melhor_acc - media < 5:
        print("AVISO: esse pico mal se destaca da media "
              f"({media:.1f}%). Provavelmente ainda nao ha dados suficientes "
              "ou a API e a cabine nao se correspondem como esperado.")


if __name__ == "__main__":
    main()
