"""
Aprende o limiar de cada gancho a partir dos dados ja coletados.

O limiar unico (4.5) veio de uma medicao boa, mas media todos os ganchos. Cada
gancho tem uma situacao diferente: os da frente ficam contra parede lisa e
separam bem; os do fundo caem numa regiao densa de correntes e pontuam alto
mesmo vazios. Um limiar por gancho corrige isso.

LIMITACAO SERIA DO GABARITO - leia antes de confiar no resultado:

    A API so lista pecas que tem programa de robo de pintura. Uma peca sem
    programa fica pendurada no gancho e NAO aparece com posicao (chega a
    aparecer na lista com o campo hook vazio). Ou seja:

        API diz OCUPADO  -> confiavel, tem peca ali
        API diz vazio    -> NAO confiavel, pode ser peca sem programa

    Como o aprendizado precisa de exemplos das duas classes, ele acaba usando
    "API nao listou" como se fosse vazio. Quando isso estiver errado, o limiar
    aprendido fica alto demais - o gancho tinha peca, o script achou que era
    fundo, e conclui que aquele nivel de score significa vazio.

    Consequencia pratica: um gancho reprovado aqui por "sobreposicao" pode nao
    ser um gancho ilegivel, e sim um gancho que costuma receber peca sem
    programa. Trate o resultado como indicio, nao veredito.

    Para resolver de verdade seria preciso rotulo humano nos casos vazios, ou
    subtracao de fundo (foto da cabine sem carro), que torna o sinal menos
    ambiguo.

Metrica: acerto BALANCEADO - media entre acerto nos ocupados e nos vazios.
Balanceado porque as classes sao desiguais: um gancho quase sempre vazio teria
90% de acerto so respondendo "vazio" sempre, e isso e sorte, nao deteccao.

Quando os scores das duas classes se sobrepoem, NAO existe limiar que resolva, e
o script diz isso em vez de escolher o "menos pior". Esses ganchos ficam de fora
do arquivo e continuam sendo mostrados como INCERTOS na tela.

Uso:
    python local_cv/learn_thresholds.py                 # so relatorio
    python local_cv/learn_thresholds.py --gravar        # aplica o aprendizado
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from calibration import CALIB_DIR, thresholds_path_for


def carregar(csv_path: str, min_confianca: float) -> dict[str, dict[int, list[tuple[float, int]]]]:
    """Le o CSV e agrupa (score, rotulo da API) por calibracao e por gancho.

    Descarta duas coisas que envenenariam o aprendizado:
      - linhas em que a API nao respondeu: a lista de ocupados vem vazia e usar
        isso como gabarito ensinaria "tudo vazio";
      - capturas com confianca de calibracao baixa: e o carro entrando ou
        saindo, os pontos nem estao sobre os ganchos, entao o score nao
        descreve o gancho nenhum.
    """
    if not os.path.exists(csv_path):
        raise SystemExit(f"Nao encontrei {csv_path}. Rode local_cv/collect_validation.py antes.")

    dados: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    sem_gabarito = em_transicao = 0

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        if "api_ok" not in (leitor.fieldnames or []):
            raise SystemExit(
                f"{csv_path} esta no formato antigo, sem as colunas api_ok/ocupado_api.\n"
                "Colete novos dados com a versao atual do collect_validation.py.")

        for linha in leitor:
            if linha["api_ok"] != "1" or linha["ocupado_api"] == "":
                sem_gabarito += 1
                continue
            if float(linha["confianca_calib"]) < min_confianca:
                em_transicao += 1
                continue
            score = float(linha["score_mediana"] or linha["score"])
            dados[linha["calibracao"]][int(linha["gancho"])].append(
                (score, int(linha["ocupado_api"])))

    if sem_gabarito:
        print(f"{sem_gabarito} linha(s) sem gabarito da API foram ignoradas.")
    if em_transicao:
        print(f"{em_transicao} linha(s) com confianca < {min_confianca} (cena em "
              f"transicao) foram ignoradas.")
    if sem_gabarito or em_transicao:
        print()
    return dados


def acerto_balanceado(amostras: list[tuple[float, int]], limiar: float) -> float:
    ocupados = [s for s, rot in amostras if rot == 1]
    vazios = [s for s, rot in amostras if rot == 0]
    if not ocupados or not vazios:
        return 0.0

    tpr = sum(1 for s in ocupados if s > limiar) / len(ocupados)
    tnr = sum(1 for s in vazios if s <= limiar) / len(vazios)
    return (tpr + tnr) / 2


def melhor_limiar(amostras: list[tuple[float, int]]) -> tuple[float, float]:
    """Varre os cortes possiveis e devolve (limiar, acerto balanceado)."""
    scores = sorted({s for s, _ in amostras})
    candidatos = [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    if not candidatos:
        return 0.0, 0.0

    avaliados = [(acerto_balanceado(amostras, c), c) for c in candidatos]
    acerto, limiar = max(avaliados)
    return limiar, acerto


def analisar(gancho: int, amostras: list, limiar_atual: float,
              min_amostras: int, min_acerto: float) -> dict:
    ocupados = [s for s, rot in amostras if rot == 1]
    vazios = [s for s, rot in amostras if rot == 0]

    info = {
        "gancho": gancho,
        "n_ocupado": len(ocupados),
        "n_vazio": len(vazios),
        "med_ocupado": float(np.median(ocupados)) if ocupados else None,
        "med_vazio": float(np.median(vazios)) if vazios else None,
        "acerto_atual": acerto_balanceado(amostras, limiar_atual),
        "limiar": None,
        "acerto": 0.0,
        "motivo": None,
    }

    if len(ocupados) < min_amostras or len(vazios) < min_amostras:
        info["motivo"] = (f"dados insuficientes ({len(ocupados)} ocupado / "
                          f"{len(vazios)} vazio, minimo {min_amostras} de cada)")
        return info

    limiar, acerto = melhor_limiar(amostras)
    info["limiar"], info["acerto"] = limiar, acerto

    if acerto < min_acerto:
        info["motivo"] = (f"scores se sobrepoem demais, nenhum limiar passa de "
                          f"{acerto * 100:.0f}% - esse gancho nao e legivel assim")
        info["limiar"] = None

    return info


def relatar(calib: str, infos: list[dict], limiar_atual: float) -> None:
    print(f"\n=== {calib} ===")
    print(f"{'g':>3}  {'n_ocup':>6} {'n_vaz':>6}  {'med_ocup':>8} {'med_vaz':>8}  "
          f"{'atual':>6}  {'limiar':>6} {'acerto':>7}")

    for i in infos:
        med_o = f"{i['med_ocupado']:.1f}" if i["med_ocupado"] is not None else "-"
        med_v = f"{i['med_vazio']:.1f}" if i["med_vazio"] is not None else "-"
        atual = f"{i['acerto_atual'] * 100:.0f}%"

        if i["limiar"] is None:
            print(f"{i['gancho']:>3}  {i['n_ocupado']:>6} {i['n_vazio']:>6}  "
                  f"{med_o:>8} {med_v:>8}  {atual:>6}       -       -   <- {i['motivo']}")
        else:
            ganho = (i["acerto"] - i["acerto_atual"]) * 100
            seta = f"  ({ganho:+.0f} p.p.)" if abs(ganho) >= 0.5 else ""
            print(f"{i['gancho']:>3}  {i['n_ocupado']:>6} {i['n_vazio']:>6}  "
                  f"{med_o:>8} {med_v:>8}  {atual:>6}  "
                  f"{i['limiar']:>6.1f} {i['acerto'] * 100:>6.0f}%{seta}")

    aprovados = [i for i in infos if i["limiar"] is not None]
    print(f"\nlimiar geral em uso: {limiar_atual}")
    print(f"{len(aprovados)} de {len(infos)} ganchos ficaram legiveis.")

    reprovados = [i for i in infos if i["limiar"] is None]
    if reprovados:
        print("ganchos sem limiar confiavel (continuam marcados INCERTO na tela): "
              f"{[i['gancho'] for i in reprovados]}")


def gravar(calib: str, infos: list[dict], total_amostras: int) -> str:
    caminho = thresholds_path_for(os.path.join(CALIB_DIR, calib))
    conteudo = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "calibracao": calib,
        "amostras": total_amostras,
        "limiares": {str(i["gancho"]): round(i["limiar"], 2)
                      for i in infos if i["limiar"] is not None},
        "sem_limiar": {str(i["gancho"]): i["motivo"]
                        for i in infos if i["limiar"] is None},
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(conteudo, f, indent=2, ensure_ascii=True)
    return caminho


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aprende o limiar de cada gancho a partir do CSV de validacao")
    parser.add_argument("--csv", default="logs/validacao_ganchos.csv")
    parser.add_argument("--limiar-atual", type=float, default=4.5,
                         help="Limiar geral, usado como base de comparacao")
    parser.add_argument("--min-amostras", type=int, default=10,
                         help="Minimo de exemplos de CADA classe por gancho")
    parser.add_argument("--min-acerto", type=float, default=0.85,
                         help="Acerto balanceado minimo para aceitar um limiar")
    parser.add_argument("--min-confianca", type=float, default=10.0,
                         help="Ignora capturas com a cena em transicao")
    parser.add_argument("--gravar", action="store_true",
                         help="Grava os limiares aprendidos (padrao: so relatorio)")
    args = parser.parse_args()

    dados = carregar(args.csv, args.min_confianca)
    if not dados:
        raise SystemExit("Nenhuma linha com gabarito da API. Colete mais dados.")

    print("ATENCAO: a API so lista pecas com programa de robo. Um gancho com peca")
    print("sem programa aparece aqui como 'vazio', o que empurra o limiar para")
    print("cima. Trate os limiares abaixo como indicio, nao como verdade.")

    for calib, por_gancho in sorted(dados.items()):
        infos = [analisar(g, por_gancho[g], args.limiar_atual,
                           args.min_amostras, args.min_acerto)
                  for g in sorted(por_gancho)]
        relatar(calib, infos, args.limiar_atual)

        if args.gravar:
            total = sum(len(v) for v in por_gancho.values())
            print(f"gravado: {gravar(calib, infos, total)}")

    if not args.gravar:
        print("\nRelatorio apenas. Rode com --gravar para a deteccao passar a usar isso.")


if __name__ == "__main__":
    main()
