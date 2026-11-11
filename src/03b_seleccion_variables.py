"""
03b - Selección de variables por eliminación hacia atrás, SOLO con 2023-2024 y mora 30+ a 6 meses (protocolo, sección 4).

Parte de las 17 variables permitidas. Modelo de tamizaje: LightGBM con parámetros fijos (el mismo de la revisión
anterior), con la ciudad y el departamento codificados por la tasa del evento dentro del pliegue. Métrica: AUC dentro
de cada cosecha, en CV 5x5 estratificada por año x evento.
Aporte de una variable = AUC con todas - AUC sin ella (promedio de los 25 pliegues). Se quita la de menor aporte
mientras ese aporte sea <= 0 (quitarla no baja el AUC promedio). Se detiene cuando todas aportan algo positivo.

Salidas: resultados/protocolo/seleccion_variables.json (variables elegidas) y seleccion_camino.csv (cada paso).
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import protocolo as P
import utilidades_auc as U

Y = P.Y
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, n_jobs=4)
ALTA = ["ciudad", "departamento"]


def te(xtr, ytr, xte, m=20):
    prior = ytr.mean()
    g = pd.DataFrame({"x": xtr.values, "y": ytr.values}).groupby("x")["y"].agg(["sum", "count"])
    return xte.map((g["sum"] + m * prior) / (g["count"] + m)).fillna(prior).values


def te_oof(x, y):
    out = np.zeros(len(x))
    for a, c in StratifiedKFold(5, shuffle=True, random_state=99).split(x, y):
        out[c] = te(x.iloc[a], y.iloc[a], x.iloc[c])
    return out


def matrices(d, cols, tr, va, y):
    A, B = pd.DataFrame(index=d.index[tr]), pd.DataFrame(index=d.index[va])
    for c in cols:
        if c in ALTA:
            A[c] = te_oof(d[c].iloc[tr], y.iloc[tr])
            B[c] = te(d[c].iloc[tr], y.iloc[tr], d[c].iloc[va])
        elif d[c].dtype == object:
            cats = pd.Categorical(d[c]).categories
            A[c] = pd.Categorical(d[c].iloc[tr], categories=cats)
            B[c] = pd.Categorical(d[c].iloc[va], categories=cats)
        else:
            A[c], B[c] = d[c].iloc[tr].values, d[c].iloc[va].values
    return A, B


def evaluar(d, cols, y, anio, folds):
    """AUC dentro de cosecha en cada uno de los 25 pliegues."""
    res = []
    for semilla, lista in folds:
        for tr, va in lista:
            A, B = matrices(d, cols, tr, va, y)
            m = lgb.LGBMClassifier(random_state=semilla, **PARAMS).fit(A, y.iloc[tr])
            res.append(P.auc_intra(y.iloc[va], m.predict_proba(B)[:, 1], anio[va]))
    return np.array(res)


def main():
    P.SALIDA.mkdir(parents=True, exist_ok=True)
    dev = P.cargar_desarrollo()
    d = dev[dev[Y].notna()].reset_index(drop=True)
    y, anio = d[Y].astype(int), d["anio"].values
    folds = [(s, P.pliegues(anio, y, s)) for s in P.SEMILLAS_EVAL]
    actuales = list(P.PERMITIDAS)
    camino, paso = [], 0
    t0 = time.time()
    while len(actuales) > 1:
        base = evaluar(d, actuales, y, anio, folds)
        aportes = {}
        for v in actuales:
            sin = evaluar(d, [c for c in actuales if c != v], y, anio, folds)
            difs = base - sin
            media, ee, inf, sup, p = U.nadeau_bengio(difs)
            aportes[v] = media
            camino.append({"paso": paso, "n_variables": len(actuales), "auc_con_todas": base.mean(), "variable": v,
                           "aporte": media, "aporte_ee_corregido": ee, "ic_inf": inf, "ic_sup": sup,
                           "pliegues_aporte_positivo": int((difs > 0).sum())})
        peor = min(aportes, key=aportes.get)
        print(f"paso {paso}: {len(actuales)} variables, AUC {base.mean():.4f}; menor aporte {peor} "
              f"{aportes[peor]:+.4f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(camino).to_csv(P.SALIDA / "seleccion_camino.csv", index=False, encoding="utf-8-sig")
        if aportes[peor] > 0:
            break
        actuales.remove(peor)
        camino.append({"paso": paso, "n_variables": len(actuales) + 1, "variable": peor, "eliminada": True})
        paso += 1
    final = evaluar(d, actuales, y, anio, folds)
    resultado = {"variables": actuales, "auc_intra_cosecha": float(final.mean()), "auc_de_pliegues": float(final.std()),
                 "regla": "eliminación hacia atrás; se quita la de menor aporte mientras sea <= 0",
                 "modelo_tamizaje": "LightGBM parámetros fijos", "datos": f"desembolsos 2023-2024, mora30_6m etiqueta {P.ETIQUETA}"}
    (P.SALIDA / "seleccion_variables.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(camino).to_csv(P.SALIDA / "seleccion_camino.csv", index=False, encoding="utf-8-sig")
    print("\nSeleccionadas:", actuales, f"AUC {final.mean():.4f}")


if __name__ == "__main__":
    main()
