"""
06c - Cálculo de los días de mora y señales de cambios en la cobranza.

Con el panel mensual de la población de estudio (registros posteriores al desembolso):
  1. Consistencia del cálculo de los días de mora por año del corte:
     - cuotas en mora frente a días de mora / 30 (coinciden con una cuota de tolerancia);
     - aumento mensual de los días de mora cuando la mora continúa (mediana).
  2. Señales de cobranza por año de desembolso, en los primeros 6 meses:
     - impago de la primera cuota (ocurre antes de cualquier gestión de cobro);
     - rodamiento de 1-29 días a 30 o más el mes siguiente;
     - curación: créditos que llegan a 30+ y vuelven a 0 días en los 3 meses siguientes (con 3 meses observados);
     - cobro jurídico en algún corte.
Descriptivo: no usa el modelo. Salidas en resultados/protocolo/mora_cobranza/.
"""
import numpy as np
import pandas as pd

import limpieza as L
import protocolo as P

SALIDA = P.SALIDA / "mora_cobranza"


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(P.RAIZ / "datos" / "cartera.parquet")
    panel, c0, embudo, ultimo = L.filtrar_poblacion(df)
    p = panel[["Numero_Encriptado", "Mes_Cosecha", "Corte", "Dias Mora", "Cuotas Mora", "Cobro Juridico",
               "Fecha desembolso"]].copy()
    p["corte"] = L.fecha_excel(p["Corte"])
    p["anio_des"] = L.fecha_excel(p["Fecha desembolso"]).dt.year
    p["anio_corte"] = p["corte"].dt.year
    p = p.sort_values(["Numero_Encriptado", "Mes_Cosecha"]).reset_index(drop=True)

    # 1) consistencia del cálculo
    m = p[p["Dias Mora"] > 0]
    coincide = ((m["Cuotas Mora"] - np.ceil(m["Dias Mora"] / 30)).abs() <= 1)
    g = p.groupby("Numero_Encriptado")
    p["dm_ant"], p["mob_ant"] = g["Dias Mora"].shift(), g["Mes_Cosecha"].shift()
    sigue = p[(p.mob_ant == p.Mes_Cosecha - 1) & (p.dm_ant > 0) & (p["Dias Mora"] > p.dm_ant)]
    calculo = pd.DataFrame({
        "registros_en_mora": m.groupby("anio_corte").size(),
        "pct_cuotas_coherentes": coincide.groupby(m["anio_corte"]).mean(),
        "meses_con_mora_que_sigue": sigue.groupby("anio_corte").size(),
        "aumento_mensual_mediano": (sigue["Dias Mora"] - sigue.dm_ant).groupby(sigue["anio_corte"]).median()})
    calculo.index.name = "anio_corte"
    calculo.to_csv(SALIDA / "calculo_dias_mora.csv", encoding="utf-8-sig")

    # 2) señales de cobranza por año de desembolso
    b = pd.read_parquet(P.RAIZ / "datos" / "base_modelo.parquet")
    b["anio"] = b["fecha_desembolso"].dt.year
    primera = b.groupby("anio").no_pago_primera_cuota.agg(["count", "mean"])

    seis = p[p.Mes_Cosecha.between(1, 6)].copy()
    gs = seis.groupby("Numero_Encriptado")
    seis["dm_sig"], seis["mob_sig"] = gs["Dias Mora"].shift(-1), gs["Mes_Cosecha"].shift(-1)
    r = seis[(seis.mob_sig == seis.Mes_Cosecha + 1) & seis["Dias Mora"].between(1, 29)]
    rodamiento = (r.dm_sig >= 30).groupby(r.anio_des).agg(["count", "mean"])

    mob30 = p[p.Mes_Cosecha.between(1, 6) & (p["Dias Mora"] >= 30)].groupby("Numero_Encriptado").Mes_Cosecha.min().rename("mob30")
    q = p.merge(mob30, left_on="Numero_Encriptado", right_index=True)
    v = q[(q.Mes_Cosecha > q.mob30) & (q.Mes_Cosecha <= q.mob30 + 3)]
    cur = pd.DataFrame({"meses": v.groupby("Numero_Encriptado").Mes_Cosecha.count(),
                        "cura": v.groupby("Numero_Encriptado")["Dias Mora"].min() == 0,
                        "anio": q.groupby("Numero_Encriptado").anio_des.first()}).dropna()
    cur = cur[cur.meses == 3]
    curacion = cur.groupby("anio").cura.agg(["count", "mean"])

    juridico = p.assign(j=p["Cobro Juridico"].astype(str).str.upper().eq("SI")).groupby("Numero_Encriptado") \
        .agg(anio=("anio_des", "first"), j=("j", "max")).groupby("anio").j.agg(["count", "mean"])

    cobranza = pd.concat({"impago_primera_cuota": primera, "rodamiento_1_29_a_30": rodamiento,
                          "curacion_tras_30": curacion, "cobro_juridico": juridico}, axis=1)
    cobranza.columns = [f"{a}__{'n' if c == 'count' else 'tasa'}" for a, c in cobranza.columns]
    cobranza.index.name = "anio_desembolso"
    cobranza.to_csv(SALIDA / "senales_cobranza.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print(calculo.round(3).to_string()); print(cobranza.round(3).to_string())


if __name__ == "__main__":
    main()
