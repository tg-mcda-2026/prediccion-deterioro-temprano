"""
06b - Caída de la mora entre cosechas: tabla por trimestre de desembolso (independientes, motos, C-0) con la mora y las
condiciones de aprobación que muestra la base, más los pagos anticipados (salidas del panel sin deterioro) por
cosecha y la tasa del evento por año.

No entra al modelo: describe la población para explicar la caída de la mora entre cosechas.
Salidas: resultados/protocolo/politica_cosechas.csv, salidas_por_cosecha.csv, tasa_por_regla.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import limpieza as L
import protocolo as P

RAIZ = Path(__file__).resolve().parents[1]


def main():
    base = pd.read_parquet(RAIZ / "datos" / "base_protocolo.parquet")
    df = pd.read_parquet(RAIZ / "datos" / "cartera.parquet")
    c0 = df[df["Mes_Cosecha"] == 0].drop_duplicates("Numero_Encriptado").set_index("Numero_Encriptado").loc[base["id_obligacion"]]
    b = base.set_index("id_obligacion")
    t = pd.DataFrame(index=b.index)
    t["trimestre"] = b["fecha_desembolso"].dt.to_period("Q").astype(str)
    t["mora30_6m"] = b[P.Y]  # etiqueta principal
    t["mora30_3m"] = b[f"mora30_3m__{P.ETIQUETA}"]
    t["prenda"] = (c0["Prenda"] == "SI").astype(float).values
    t["cuota_inicial_pct"] = c0["%Initial_fee"].values
    t["plazo"] = c0["Time_limit"].values
    t["plazo_48_mas"] = (c0["Time_limit"] >= 48).astype(float).values
    t["plazo_12_menos"] = (c0["Time_limit"] <= 12).astype(float).values
    t["tasa_mensual"] = c0["Interest_Rate"].values
    t["aprobacion_automatica"] = (L.normalizar_texto(c0["Approval"]) == "AUTOMATICO").astype(float).values
    t["segmento_oro"] = (L.normalizar_texto(c0["Segment"]) == "ORO").astype(float).values
    t["informal"] = b["independiente_informal"]
    # participación de la familia más frecuente entre los scooters (Scooter A); por confidencialidad el código no
    # nombra los modelos (limpieza.etiquetas_familia)
    scooter_a = next(f for f, e in L.etiquetas_familia(base).items() if e == "Scooter A")
    t["scooter_a"] = (b["familia_moto"] == scooter_a).astype(float)
    pol = t.groupby("trimestre").agg(creditos=("prenda", "size"), mora30_6m=("mora30_6m", "mean"),
                                     con_resultado_6m=("mora30_6m", "count"), mora30_3m=("mora30_3m", "mean"),
                                     prenda=("prenda", "mean"), cuota_inicial_pct=("cuota_inicial_pct", "mean"),
                                     plazo_medio=("plazo", "mean"), plazo_48_mas=("plazo_48_mas", "mean"),
                                     plazo_12_menos=("plazo_12_menos", "mean"), tasa_mensual=("tasa_mensual", "mean"),
                                     aprobacion_automatica=("aprobacion_automatica", "mean"),
                                     segmento_oro=("segmento_oro", "mean"), informal=("informal", "mean"),
                                     scooter_a=("scooter_a", "mean"))
    pol.round(4).to_csv(P.SALIDA / "politica_cosechas.csv", encoding="utf-8-sig")

    sal = []
    for h in (3, 6):
        madura = b[f"estado_{h}m"] != "inmaduro"
        sin_det = b[f"mora30_{h}m"].isna() & madura
        g = pd.DataFrame({"anio": b["anio"], "madura": madura, "sale": sin_det,
                          "mes0": sin_det & (b[f"estado_{h}m"] == "sale_sin_exposicion")})
        r = g[g.madura].groupby("anio").agg(maduros=("sale", "size"), salen_sin_deterioro=("sale", "sum"),
                                            salen_mes0=("mes0", "sum"))
        r["pct_salen"] = r.salen_sin_deterioro / r.maduros
        ct = pd.crosstab(g[g.madura].anio, g[g.madura].sale)
        r["chi2_p_entre_anios"] = chi2_contingency(ct)[1]
        sal.append(r.assign(ventana=f"{h} meses").reset_index())
    pd.concat(sal).to_csv(P.SALIDA / "salidas_por_cosecha.csv", index=False, encoding="utf-8-sig")

    tasas = []
    for e in P.ETIQUETAS:
        tasas.append(b.groupby("anio")[f"mora30_6m__{e}"].agg(["count", "mean"]).assign(regla=e).reset_index())
    pd.concat(tasas).to_csv(P.SALIDA / "tasa_por_regla.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(pol.round(3).to_string())
    print(pd.concat(sal).round(3).to_string(index=False))
    print(pd.concat(tasas).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
