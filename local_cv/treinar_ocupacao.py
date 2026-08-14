"""Treina o classificador de ocupacao (ocupado/vazio) a partir do dataset
coletado ao vivo no monitor_screen.py.

Como o dataset cresce: no monitor, tecla 'l' entra em modo rotulo (pausa o
video); clique numa linha do gancho no painel INVERTE a leitura daquele gancho
se o detector/modelo errou; tecla 's' salva os recortes do frame atual (com a
leitura corrigida onde voce clicou, e a leitura normal onde nao clicou) em
dataset_ocupacao/ocupado/ e dataset_ocupacao/vazio/.

100% CPU, sem CUDA: features de grade (intensidade + densidade de borda por
celula, ver local_cv/modelo_ocupacao.py) + RandomForest (scikit-learn).

Uso:
    python local_cv/treinar_ocupacao.py
    python local_cv/treinar_ocupacao.py --dataset dataset_ocupacao --min-exemplos 40
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from local_cv.modelo_ocupacao import CAMINHO_MODELO_PADRAO, features


def carregar_dataset(pasta: str):
    X, y = [], []
    for rotulo, sub in ((1, "ocupado"), (0, "vazio")):
        for caminho in glob.glob(os.path.join(pasta, sub, "*.jpg")):
            img = cv2.imread(caminho)
            if img is None:
                continue
            X.append(features(img))
            y.append(rotulo)
    return np.array(X), np.array(y)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset_ocupacao")
    p.add_argument("--out", default=CAMINHO_MODELO_PADRAO)
    p.add_argument("--min-exemplos", type=int, default=40,
                   help="Minimo de recortes por classe pra treinar algo que preste")
    args = p.parse_args()

    if not os.path.isdir(args.dataset):
        raise SystemExit(f"{args.dataset} nao existe ainda - rotule no monitor_screen.py primeiro")

    X, y = carregar_dataset(args.dataset)
    n_ocupado, n_vazio = int((y == 1).sum()) if len(y) else 0, int((y == 0).sum()) if len(y) else 0
    print(f"exemplos: ocupado={n_ocupado}  vazio={n_vazio}")
    if n_ocupado < args.min_exemplos or n_vazio < args.min_exemplos:
        raise SystemExit(
            f"pouco dado ainda (minimo {args.min_exemplos} por classe). "
            f"Continue rotulando no monitor ('l' + clique nas linhas erradas + 's') "
            f"antes de treinar.")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import joblib

    modelo = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
    scores = cross_val_score(modelo, X, y, cv=5, scoring="balanced_accuracy")
    print(f"acerto balanceado (5-fold, dado de treino): {scores.mean():.2f} +- {scores.std():.2f}")

    modelo.fit(X, y)
    joblib.dump(modelo, args.out)
    print(f"modelo salvo em {args.out}")


if __name__ == "__main__":
    main()
