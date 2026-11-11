"""
Condiciones del crédito en el mes de desembolso (registro C-0). NO entran a la versión elegida (C3): se usan solo para
contrastar la primera hipótesis (bloque financiero-contractual), para describir la población y para la Fase 6
(capital e ingreso financiero contratado).
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]

FINANCIERAS = ["capital", "cuota", "cuota_inicial", "cuota_inicial_pct", "tasa", "plazo", "prenda"]


def variables_credito(ids):
    """Capital, cuota, cuota inicial (pesos y %), tasa mensual, plazo y prenda del registro C-0 de cada obligación."""
    df = pd.read_parquet(RAIZ / "datos" / "cartera.parquet",
                         columns=["Numero_Encriptado", "Mes_Cosecha", "Valor Capital", "Valor Cuota", "CUOTA_INICIAL",
                                  "%Initial_fee", "Interest_Rate", "Time_limit", "Prenda"])
    c0 = df[df["Mes_Cosecha"] == 0].drop_duplicates("Numero_Encriptado").set_index("Numero_Encriptado").loc[list(ids)]
    out = pd.DataFrame(index=c0.index)
    out["capital"] = c0["Valor Capital"].astype(float)
    out["cuota"] = c0["Valor Cuota"].astype(float)
    out["cuota_inicial"] = c0["CUOTA_INICIAL"].astype(float)
    out["cuota_inicial_pct"] = c0["%Initial_fee"].astype(float)
    out["tasa"] = c0["Interest_Rate"].astype(float)
    out["plazo"] = c0["Time_limit"].astype(float)
    out["prenda"] = (c0["Prenda"].astype(str).str.strip().str.upper() == "SI").astype(float)
    # Ingreso financiero contratado aproximado: lo que suman las cuotas menos el capital. La cuota puede incluir seguros,
    # así que es una aproximación y se declara como tal.
    out["ingreso_contratado"] = (out["cuota"] * out["plazo"] - out["capital"]).clip(lower=0)
    out.index.name = "id_obligacion"
    return out
