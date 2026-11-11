"""Inferencia para AUC y tasas: DeLong con IC de la diferencia, bootstrap pareado, Nadeau-Bengio, Wilson, Holm."""
import numpy as np
from scipy.stats import norm, t as t_student
from sklearn.metrics import roc_auc_score


def _midrank(x):
    orden = np.argsort(x)
    z = x[orden]
    r = np.zeros(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and z[j] == z[i]:
            j += 1
        r[i:j] = 0.5 * (i + j - 1)
        i = j
    salida = np.empty(len(x))
    salida[orden] = r + 1
    return salida


def delong(y, p1, p2):
    """AUC de dos puntajes sobre los mismos casos: (auc1, auc2, diferencia, error estándar, IC95 inf, IC95 sup, p)."""
    y = np.asarray(y)
    orden = np.argsort(-y, kind="stable")
    m = int(y.sum())
    n = len(y) - m
    preds = np.vstack([p1, p2])[:, orden]
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    ee = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    dif = float(aucs[0] - aucs[1])
    p = float(2 * norm.sf(abs(dif) / ee)) if ee > 0 else 1.0
    return float(aucs[0]), float(aucs[1]), dif, ee, dif - 1.96 * ee, dif + 1.96 * ee, p


def ic_auc_delong(y, p):
    """IC95 de un AUC por DeLong."""
    y = np.asarray(y)
    orden = np.argsort(-y, kind="stable")
    m = int(y.sum())
    n = len(y) - m
    x = np.asarray(p)[orden]
    tx, ty, tz = _midrank(x[:m]), _midrank(x[m:]), _midrank(x)
    auc = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    ee = np.sqrt(np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n)
    return float(auc), float(auc - 1.96 * ee), float(auc + 1.96 * ee)


def bootstrap_pareado(y, p1, p2, n=2000, semilla=0):
    """IC95 de la diferencia de AUC por bootstrap estratificado (remuestrea eventos y no eventos por separado)."""
    rng = np.random.default_rng(semilla)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    difs = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        difs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return tuple(np.percentile(difs, [2.5, 97.5]))


def nadeau_bengio(difs, frac_prueba=0.2):
    """t corregida para diferencias pareadas por pliegue en validación cruzada repetida (pliegues solapados).
    Devuelve (media, error estándar corregido, IC95 inf, IC95 sup, p)."""
    d = np.asarray(difs, dtype=float)
    J = len(d)
    var = np.var(d, ddof=1) * (1 / J + frac_prueba / (1 - frac_prueba))
    ee = float(np.sqrt(var))
    tcrit = t_student.ppf(0.975, J - 1)
    media = float(d.mean())
    p = float(2 * t_student.sf(abs(media) / ee, J - 1)) if ee > 0 else 1.0
    return media, ee, media - tcrit * ee, media + tcrit * ee, p


def wilson(exitos, n, z=1.96):
    """IC de Wilson para una proporción."""
    if n == 0:
        return np.nan, np.nan
    phat = exitos / n
    centro = (phat + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    radio = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return float(centro - radio), float(centro + radio)


def holm(pvalores):
    """Valores p ajustados por Holm (en el mismo orden de entrada)."""
    p = np.asarray(pvalores, dtype=float)
    orden = np.argsort(p)
    k = len(p)
    ajust = np.empty(k)
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (k - rango) * p[i]))
        ajust[i] = acumulado
    return ajust
