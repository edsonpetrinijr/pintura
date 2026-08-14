"""Classificador treinavel de ocupacao do gancho (ocupado/vazio).

Complementa (nao substitui ainda) a heuristica de bordas do detect_hooks_local.
Roda 100% em CPU (notebook sem CUDA): features simples de grade (intensidade +
densidade de borda por celula, mesmo Canny 50/150 do detect_hooks_local) +
RandomForest (scikit-learn). Nao usa cv2.HOGDescriptor porque o build de
opencv-python instalado aqui (5.0.0) nao expoe essa classe no binding python.
Features de tamanho fixo (resize pro canonico antes de tudo) para aceitar
janelas de qualquer tamanho/proporcao sem re-treinar o extrator.

O modelo so existe depois que alguem rodar treinar_ocupacao.py sobre o dataset
coletado ao vivo no monitor_screen.py (tecla 'l' entra em modo rotulo, clique
numa linha do gancho corrige a leitura, 's' salva os recortes daquele frame).
"""
import os

import cv2
import numpy as np

TAMANHO = (64, 64)  # resize antes de tudo - cancela a variacao de tamanho da janela
GRID = 8  # celulas de 8x8 px dentro do TAMANHO

CAMINHO_MODELO_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "modelo_ocupacao.joblib")


def features(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("crop vazio")
    cinza = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    cinza = cv2.resize(cinza, TAMANHO)
    bordas = cv2.Canny(cv2.GaussianBlur(cinza, (5, 5), 0), 50, 150)
    celula = TAMANHO[0] // GRID
    vals = []
    for i in range(GRID):
        for j in range(GRID):
            bloco_cinza = cinza[i * celula:(i + 1) * celula, j * celula:(j + 1) * celula]
            bloco_borda = bordas[i * celula:(i + 1) * celula, j * celula:(j + 1) * celula]
            vals.append(bloco_cinza.mean() / 255.0)
            vals.append(bloco_borda.mean() / 255.0)
    return np.array(vals, dtype=np.float32)


def carregar_modelo(caminho: str = CAMINHO_MODELO_PADRAO):
    """None se ainda nao foi treinado - quem chama deve tratar esse caso."""
    if not os.path.exists(caminho):
        return None
    import joblib
    return joblib.load(caminho)


def prever(modelo, crop_bgr: np.ndarray):
    """(ocupado: bool, confianca 0..1) ou None se modelo/crop invalido."""
    if modelo is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    x = features(crop_bgr).reshape(1, -1)
    prob = modelo.predict_proba(x)[0]
    classes = list(modelo.classes_)
    p_ocupado = prob[classes.index(1)]
    return p_ocupado >= 0.5, float(max(p_ocupado, 1 - p_ocupado))
