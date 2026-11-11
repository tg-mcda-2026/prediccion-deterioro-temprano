"""
04b - Decisión de la versión con la regla registrada (protocolo, sección 6). Solo usa la CV de 2023-2024.

  1. AUC dentro de cosecha del ensamble en cada uno de los 25 pliegues (5 semillas x 5 pliegues).
  2. Mejor = mayor AUC promedio.
  3. Diferencia pareada de cada versión frente a la mejor, pliegue a pliegue, con t corregida de Nadeau-Bengio.
     No distinguible de la mejor = el IC95 corregido incluye el 0.
  4. Se elige la más simple entre las no distinguibles de la mejor (protocolo.simplicidad).
Se hace con la etiqueta del evento (protocolo.ETIQUETA: pagos anticipados como buenos pagadores). También la ablación
(cuánto se pierde al quitar el sexo de C3) y la t de cada diferencia.

Escribe resultados/protocolo/decision.json (con la versión, sus variables y la fecha) y tablas de soporte.
"""
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import protocolo as P
import utilidades_auc as U

OOF = P.SALIDA / "oof"


def aucs_por_pliegue(version, etiqueta, modelo="ensamble"):
    d = pd.read_parquet(OOF / f"{version}__{etiqueta}.parquet")
    d = d[d.modelo == modelo]
    return d.groupby(["semilla", "pliegue"]).apply(lambda g: P.auc_intra(g.y, g.p, g.anio)).sort_index()


def tabla(versiones, etiqueta):
    A = pd.DataFrame({v: aucs_por_pliegue(v, etiqueta) for v in versiones})
    mejor = A.mean().idxmax()
    filas = []
    for v in versiones:
        media, ee, inf, sup, p = U.nadeau_bengio((A[v] - A[mejor]).values) if v != mejor else (0, 0, 0, 0, 1)
        filas.append({"etiqueta": etiqueta, "version": v, "n_variables": len(P.candidatas().get(v, P.ABLACION.get(v, []))),
                      "auc_intra_cosecha": A[v].mean(), "de_pliegues": A[v].std(),
                      "dif_vs_mejor": media, "ee_corregido": ee, "ic_inf": inf, "ic_sup": sup, "p": p,
                      "no_distinguible_de_la_mejor": bool(v == mejor or (inf <= 0 <= sup)), "es_mejor": v == mejor})
    return pd.DataFrame(filas), A


def elegir(t):
    cand = P.candidatas()
    emp = t[t.no_distinguible_de_la_mejor & t.version.isin(cand)]
    return min(emp.version, key=lambda v: P.simplicidad(cand[v]))


def main():
    cand = list(P.candidatas())
    etiquetas = [e for e in P.ETIQUETAS if all((OOF / f"{v}__{e}.parquet").exists() for v in cand)]
    assert P.ETIQUETA in etiquetas, f"faltan las corridas de la etiqueta {P.ETIQUETA}"
    resultados, elegidas = [], {}
    for e in etiquetas:
        t, A = tabla(cand, e)
        elegidas[e] = elegir(t)
        t["elegida"] = t.version == elegidas[e]
        resultados.append(t)
        A.to_csv(P.SALIDA / f"auc_pliegues__{e}.csv", encoding="utf-8-sig")
    T = pd.concat(resultados)
    T.to_csv(P.SALIDA / "decision_tabla.csv", index=False, encoding="utf-8-sig")

    # Ablación (etiqueta principal): pérdida por quitar el sexo de C3, con IC corregido
    abl = []
    if all((OOF / f"{v}__{P.ETIQUETA}.parquet").exists() for v in list(P.ABLACION) + ["C3_V2p"]):
        A = pd.DataFrame({v: aucs_por_pliegue(v, P.ETIQUETA) for v in list(P.ABLACION) + ["C3_V2p"]})
        media, ee, inf, sup, p = U.nadeau_bengio((A["C3_V2p"] - A["A_C3_con_sexo"]).values)
        abl.append({"cambio": "quitar sexo", "auc_con_sexo": A["A_C3_con_sexo"].mean(), "auc_sin": A["C3_V2p"].mean(),
                    "diferencia": media, "ic_inf": inf, "ic_sup": sup, "p": p})
        pd.DataFrame(abl).to_csv(P.SALIDA / "ablacion_sexo.csv", index=False, encoding="utf-8-sig")

    v = elegidas[P.ETIQUETA]
    decision = {"fecha": datetime.now().isoformat(timespec="seconds"), "version": v,
                "variables": P.candidatas()[v], "etiqueta_principal": P.ETIQUETA,
                "regla": "más simple entre las no distinguibles de la mejor (IC95 Nadeau-Bengio incluye 0); AUC dentro de cosecha, CV 5x5 2023-2024"}
    if "--registrar" in sys.argv:
        P.DECISION.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
        print("decision.json registrado")
    pd.set_option("display.width", 220)
    print(T.round(4).to_string(index=False))
    print(pd.DataFrame(abl).round(4).to_string(index=False))
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
