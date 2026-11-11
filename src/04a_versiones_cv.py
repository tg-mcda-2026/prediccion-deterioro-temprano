"""
04a - Versiones candidatas evaluadas SOLO con validación cruzada dentro de 2023-2024 (protocolo, secciones 2, 5 y 7).

Para cada versión y etiqueta:
  - Etiqueta principal (protocolo.ETIQUETA): Optuna (log-loss) por algoritmo con su propia CV (semilla 42, pliegues
    año x evento). Las demás etiquetas reutilizan esos hiperparámetros (así la diferencia es solo la etiqueta).
  - Validación cruzada repetida 5x5 (semillas 1 a 5, distintas de la de Optuna): predicciones fuera de pliegue de los
    7 algoritmos y del ensamble (promedio), guardadas con el pliegue de cada crédito.
Nada de 2025 se lee aquí (protocolo.cargar_desarrollo lo verifica).

Uso:   python 04a_versiones_cv.py [etiqueta] [version ...]
       sin argumentos: etiqueta principal, todas las candidatas, la ablación y la sensibilidad sin estado civil
       (revisión final; exploratoria).
Salidas: resultados/protocolo/oof/{version}__{etiqueta}.parquet, parametros.json, log
"""
from pathlib import Path
import json
import sys
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score

import protocolo as P

RAIZ = Path(__file__).resolve().parents[1]
import modelos as M
M.NUMERICAS, M.CAT_BAJA, M.CAT_ALTA = P.NUMERICAS, P.CAT_BAJA, P.CAT_ALTA
MODELOS = list(M.N_TRIALS)
OOF = P.SALIDA / "oof"
PARAMS = P.SALIDA / "parametros.json"


def ajustar(nombre, variables, X, y, folds):
    def objetivo(t):
        modelo = M.construir(nombre, M.espacio(nombre, t), variables)
        return -cross_val_score(modelo, X, y, cv=folds, scoring="neg_log_loss").mean()
    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=P.SEMILLA_OPTUNA))
    estudio.optimize(objetivo, n_trials=M.N_TRIALS[nombre])
    return estudio.best_params


def correr(version, variables, etiqueta, guardados):
    archivo = OOF / f"{version}__{etiqueta}.parquet"
    if archivo.exists():
        print(f"ya existe {archivo.name}", flush=True)
        return
    col = f"{P.OBJETIVO}__{etiqueta}"
    dev = P.cargar_desarrollo()
    d = dev[dev[col].notna()].reset_index(drop=True)
    y, anio = d[col].astype(int).values, d["anio"].values
    print(f"\n=== {version} [{etiqueta}] {len(variables)} variables, n={len(d)}, tasa={y.mean():.3f}", flush=True)
    filas = []
    for nombre in MODELOS:
        t0 = time.time()
        clave = f"{version}|{nombre}"
        if etiqueta == P.ETIQUETA and clave not in guardados:
            guardados[clave] = ajustar(nombre, variables, d, y, P.pliegues(anio, y, P.SEMILLA_OPTUNA))
            PARAMS.write_text(json.dumps(guardados, indent=1), encoding="utf-8")
        params = guardados[clave]
        for semilla in P.SEMILLAS_EVAL:
            folds = P.pliegues(anio, y, semilla)
            p = cross_val_predict(M.construir(nombre, params, variables), d, y, cv=folds, method="predict_proba")[:, 1]
            pliegue = np.empty(len(d), dtype=int)
            for k, (_, va) in enumerate(folds):
                pliegue[va] = k
            filas.append(pd.DataFrame({"id_obligacion": d["id_obligacion"].values, "anio": anio, "y": y,
                                       "semilla": semilla, "pliegue": pliegue, "modelo": nombre, "p": p}))
        print(f"  {nombre:13s} {time.time() - t0:6.0f}s", flush=True)
    largo = pd.concat(filas)
    ens = largo.groupby(["id_obligacion", "anio", "y", "semilla", "pliegue"], as_index=False)["p"].mean()
    ens["modelo"] = "ensamble"
    salida = pd.concat([largo, ens])
    salida.to_parquet(archivo, index=False)
    r = ens.groupby("semilla").apply(lambda g: P.auc_intra(g.y, g.p, g.anio))
    print(f"  ensamble AUC dentro de cosecha {r.mean():.4f} (DE entre semillas {r.std():.4f})", flush=True)


def main():
    OOF.mkdir(parents=True, exist_ok=True)
    etiqueta = sys.argv[1] if len(sys.argv) > 1 else P.ETIQUETA
    todas = {**P.candidatas(), **P.ABLACION, **P.SENSIBILIDAD}
    versiones = sys.argv[2:] or list(todas)
    guardados = json.loads(PARAMS.read_text(encoding="utf-8")) if PARAMS.exists() else {}
    for v in versiones:
        correr(v, todas[v], etiqueta, guardados)


if __name__ == "__main__":
    main()
