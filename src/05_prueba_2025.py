"""
05 - Reporte en 2025 (protocolo, secciones 8 y 9). Falla si no existe resultados/protocolo/decision.json.

Ajusta cada versión candidata con todo 2023-2024 (hiperparámetros de 04a_versiones_cv.py, evento de protocolo.ETIQUETA) y
califica 2025.
  - Discriminación: AUC con IC (DeLong), KS, diferencia frente a la elegida con IC (DeLong y bootstrap pareado),
    segmento actual como referencia.
  - Calibración: Brier y nivel por año (OOF 2023-2024 y 2025), sin recalibrar y recalibrando el intercepto con
    2025-T1 para evaluar 2025-T2.
  - Detección marcando el 10 % y el 20 % de mayor puntaje, con IC de Wilson.
  - Monitoreo y confirmación con el MISMO puntaje: 30+ a 3 meses, 60+ y 90+ a 6 meses.
  - Pruebas confirmatorias con Holm (sección 8 del protocolo).
  - Auditoría de equidad por sexo (elegida sin sexo frente a C3 con sexo).
  - Efecto de cosecha con 2025 incluido (nivel frente a orden).
Salidas: resultados/protocolo/reporte_2025/
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

import protocolo as P
import utilidades_auc as U

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
import modelos as M
M.NUMERICAS, M.CAT_BAJA, M.CAT_ALTA = P.NUMERICAS, P.CAT_BAJA, P.CAT_ALTA
MODELOS = list(M.N_TRIALS)
SALIDA = P.SALIDA / "reporte_2025"
RIESGO_SEGMENTO = {"BRONCE": 3, "PLATA": 2, "ORO": 1}


def ajustar_ensamble(version, variables, d, y, params):
    modelos = [M.construir(n, params[f"{version}|{n}"], variables).fit(d, y) for n in MODELOS]
    return lambda X: np.mean([m.predict_proba(X)[:, 1] for m in modelos], axis=0)


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def deteccion(y, p, frac):
    y, p = np.asarray(y), np.asarray(p)
    marcado = p >= np.quantile(p, 1 - frac)
    vp, n_ev = int((marcado & (y == 1)).sum()), int(y.sum())
    fp, n_ne = int((marcado & (y == 0)).sum()), int((y == 0).sum())
    return {"marcados": int(marcado.sum()), "eventos": n_ev, "detectados": vp, "tasa_deteccion": vp / n_ev,
            "td_ic_inf": U.wilson(vp, n_ev)[0], "td_ic_sup": U.wilson(vp, n_ev)[1],
            "falsos_positivos": fp / n_ne, "precision": vp / max(1, marcado.sum())}


def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    fit = smf.logit("y ~ lp", data=pd.DataFrame({"y": y, "lp": lp})).fit(disp=0)
    return {"tasa_observada": float(np.mean(y)), "prob_media": float(np.mean(p)), "brier": brier_score_loss(y, p),
            "brier_constante": float(np.mean(y) * (1 - np.mean(y))), "pendiente_cal": float(fit.params["lp"]),
            "intercepto_cal": float(fit.params["Intercept"])}


def main():
    decision = json.loads(P.DECISION.read_text(encoding="utf-8"))  # exige la decisión registrada
    SALIDA.mkdir(parents=True, exist_ok=True)
    elegida = decision["version"]
    params = json.loads((P.SALIDA / "parametros.json").read_text(encoding="utf-8"))
    base = P.cargar_reporte()
    dev, oot = base[base.periodo == "train"], base[base.periodo == "oot"].copy()
    versiones = {**P.candidatas(), "A_C3_con_sexo": P.ABLACION["A_C3_con_sexo"]}

    # ---------------------------------------------------------------- ajuste con 2023-2024 y calificación de 2025
    col = P.Y
    d = dev[dev[col].notna()]
    puntajes = {}
    for v, variables in versiones.items():
        puntajes[v] = ajustar_ensamble(v, variables, d, d[col].astype(int).values, params)(oot)
        print("calificado", v, flush=True)
    for v, s in puntajes.items():
        oot[f"score__{v}"] = s
    oot["score__segmento"] = oot["ref_segmento"].map(RIESGO_SEGMENTO).astype(float)
    oot.to_parquet(SALIDA / "puntajes_2025.parquet", index=False)

    # ---------------------------------------------------------------- discriminación 30+ a 6 meses (etiqueta principal)
    t = oot[oot[col].notna()]
    y = t[col].astype(int).values
    filas, difs = [], []
    for v in list(versiones) + ["segmento"]:
        s = t[f"score__{v}"].values
        m = ~np.isnan(s)
        auc, inf, sup = U.ic_auc_delong(y[m], s[m])
        fila = {"version": v, "n": int(m.sum()), "eventos": int(y[m].sum()), "auc": auc, "auc_ic_inf": inf,
                "auc_ic_sup": sup, "ks": ks(y[m], s[m])}
        if v != "segmento":
            fila.update({f"{k}_10": val for k, val in deteccion(y, s, 0.10).items()})
            fila.update({f"{k}_20": val for k, val in deteccion(y, s, 0.20).items()})
        filas.append(fila)
        if v != elegida:
            a1, a2, dif, ee, i1, i2, p = U.delong(y[m], s[m], t[f"score__{elegida}"].values[m])
            b1, b2 = U.bootstrap_pareado(y[m], s[m], t[f"score__{elegida}"].values[m], n=1000)
            difs.append({"version": v, "frente_a": elegida, "diferencia": dif, "ic_delong_inf": i1, "ic_delong_sup": i2,
                         "ic_boot_inf": b1, "ic_boot_sup": b2, "p": p})
    pd.DataFrame(filas).to_csv(SALIDA / "discriminacion_30_6m.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(difs).to_csv(SALIDA / "diferencias_vs_elegida.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- monitoreo y confirmación (mismo puntaje)
    otros = []
    for objetivo in P.OBJETIVOS_REPORTE + [P.OBJETIVO]:
        obj = f"{objetivo}__{P.ETIQUETA}"
        k = oot[obj].notna()
        for grupo, mask in [("todos", k), ("ene-jun 2025", k & (oot.fecha_desembolso < "2025-07-01")),
                            ("jul-sep 2025", k & (oot.fecha_desembolso >= "2025-07-01"))]:
            if mask.sum() == 0 or oot.loc[mask, obj].nunique() < 2:
                continue
            for v in [elegida, "A_C3_con_sexo", "segmento"]:
                s = oot.loc[mask, f"score__{v}"].values
                yy = oot.loc[mask, obj].astype(int).values
                mm = ~np.isnan(s)
                auc, inf, sup = U.ic_auc_delong(yy[mm], s[mm])
                otros.append({"objetivo": objetivo, "grupo": grupo, "version": v, "n": int(mm.sum()), "eventos": int(yy[mm].sum()),
                              "auc": auc, "ic_inf": inf, "ic_sup": sup})
    pd.DataFrame(otros).to_csv(SALIDA / "monitoreo_confirmacion.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- calibración por año y recalibración T1 -> T2
    oof = pd.read_parquet(P.SALIDA / "oof" / f"{elegida}__{P.ETIQUETA}.parquet")
    oof = oof[oof.modelo == "ensamble"].groupby(["id_obligacion", "anio", "y"], as_index=False)["p"].mean()
    cal = []
    for a in (2023, 2024):
        g = oof[oof.anio == a]
        cal.append({"periodo": str(a), "fuente": "CV (OOF)", **calib(g.y.values, g.p.values)})
    s = t[f"score__{elegida}"].values
    cal.append({"periodo": "2025", "fuente": "prueba", **calib(y, s)})
    t1 = (t.fecha_desembolso < P.CORTE_RECALIBRACION).values
    a = M.recalibrar_intercepto(y[t1], s[t1])
    cal.append({"periodo": "2025-T2", "fuente": "prueba sin recalibrar", **calib(y[~t1], s[~t1])})
    cal.append({"periodo": "2025-T2", "fuente": "prueba recalibrada con 2025-T1", "desplazamiento_logit": a,
                **calib(y[~t1], M.aplicar_intercepto(s[~t1], a))})
    pd.DataFrame(cal).to_csv(SALIDA / "calibracion_por_anio.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for etiqueta, yy, pp in [("2023 (CV)", oof[oof.anio == 2023].y, oof[oof.anio == 2023].p),
                             ("2024 (CV)", oof[oof.anio == 2024].y, oof[oof.anio == 2024].p), ("2025 (prueba)", y, s),
                             ("2025-T2 recalibrado", y[~t1], M.aplicar_intercepto(s[~t1], a))]:
        fo, mp = calibration_curve(yy, pp, n_bins=6, strategy="quantile")
        ax.plot(mp, fo, marker="o", label=etiqueta)
    ax.plot([0, 0.6], [0, 0.6], color="grey", lw=0.8)
    ax.set(xlabel="Probabilidad predicha", ylabel="Tasa observada (30+ a 6 meses)", title=f"Calibración por año - {elegida}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(SALIDA / "calibracion_por_anio.png", dpi=130); plt.close(fig)

    # ---------------------------------------------------------------- pruebas confirmatorias con Holm
    dec = pd.read_csv(P.SALIDA / "decision_tabla.csv")
    dec = dec[dec.etiqueta == P.ETIQUETA].set_index("version")
    abl = pd.read_csv(P.SALIDA / "ablacion_sexo.csv").set_index("cambio")
    seg = t["score__segmento"].values; m = ~np.isnan(seg)
    p_seg = U.delong(y[m], s[m], seg[m])[-1]
    conf = pd.DataFrame([
        {"prueba": f"{elegida} frente al segmento actual (2025)", "p": p_seg},
        {"prueba": f"{elegida} frente a la mejor en CV ({dec[dec.es_mejor].index[0]})", "p": float(dec.loc[elegida, "p"])},
        {"prueba": "pérdida por quitar el sexo (CV)", "p": float(abl.loc["quitar sexo", "p"])},
    ])
    conf["p_holm"] = U.holm(conf.p.values)
    conf.to_csv(SALIDA / "pruebas_confirmatorias.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- auditoría de equidad por sexo
    eq = []
    for v in [elegida, "A_C3_con_sexo"]:
        sc = t[f"score__{v}"].values
        umbral = {f: np.quantile(sc, 1 - f) for f in (0.10, 0.20)}
        for sexo in ["F", "M"]:
            k = (t.sexo == sexo).values
            yy, ss = y[k], sc[k]
            fila = {"version": v, "sexo": sexo, "n": int(k.sum()), "eventos": int(yy.sum()), "tasa_observada": yy.mean(),
                    "prob_media": ss.mean(), "observado_sobre_esperado": yy.mean() / ss.mean(),
                    "auc_dentro_del_sexo": roc_auc_score(yy, ss)}
            for f in (0.10, 0.20):
                marc = ss >= umbral[f]
                vp = int((marc & (yy == 1)).sum())
                fila[f"pct_marcados_{int(f*100)}"] = marc.mean()
                fila[f"tpr_{int(f*100)}"] = vp / max(1, yy.sum())
                fila[f"tpr_{int(f*100)}_ic_inf"], fila[f"tpr_{int(f*100)}_ic_sup"] = U.wilson(vp, int(yy.sum()))
                fila[f"fpr_{int(f*100)}"] = (marc & (yy == 0)).sum() / max(1, (yy == 0).sum())
            eq.append(fila)
        eq.append({"version": v, "sexo": "AUC del puntaje para separar sexos",
                   "auc_dentro_del_sexo": roc_auc_score((t.sexo == "M").astype(int), sc)})
    pd.DataFrame(eq).to_csv(SALIDA / "equidad_sexo_2025.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- efecto de cosecha con 2025 (nivel frente a orden)
    lp = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    dd = pd.concat([pd.DataFrame({"y": oof.y, "lp": lp(oof.p), "anio": oof.anio.astype(str)}),
                    pd.DataFrame({"y": y, "lp": lp(s), "anio": "2025"})])
    m0 = smf.logit("y ~ lp", dd).fit(disp=0)
    m1 = smf.logit("y ~ lp + C(anio)", dd).fit(disp=0)
    m2 = smf.logit("y ~ lp * C(anio)", dd).fit(disp=0)
    coh = pd.DataFrame([{"prueba": "nivel (efecto fijo de año)", "lr": 2 * (m1.llf - m0.llf), "gl": 2,
                         "p": chi2.sf(2 * (m1.llf - m0.llf), 2)},
                        {"prueba": "orden (puntaje x año)", "lr": 2 * (m2.llf - m1.llf), "gl": 2,
                         "p": chi2.sf(2 * (m2.llf - m1.llf), 2)}])
    coh.to_csv(SALIDA / "efecto_cosecha_con_2025.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"termino": m2.params.index, "coef": m2.params.values, "ic_inf": m2.conf_int()[0].values,
                  "ic_sup": m2.conf_int()[1].values}).to_csv(SALIDA / "efecto_cosecha_coeficientes.csv", index=False,
                                                              encoding="utf-8-sig")

    # ---------------------------------------------------------------- deciles frente al segmento y ROC
    dec_tab = pd.DataFrame({"decil": pd.qcut(s, 10, labels=range(10, 0, -1)).astype(int), "y": y, "p": s,
                            "segmento": t["ref_segmento"].values})
    res = dec_tab.groupby("decil").agg(n=("y", "size"), tasa=("y", "mean"), prob_media=("p", "mean"))
    res.join(pd.crosstab(dec_tab.decil, dec_tab.segmento, normalize="index").add_prefix("pct_")).sort_index() \
        .to_csv(SALIDA / "deciles_vs_segmento.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for v in [elegida, "A_C3_con_sexo", "segmento"]:
        sc = t[f"score__{v}"].values; mm = ~np.isnan(sc)
        fpr, tpr, _ = roc_curve(y[mm], sc[mm])
        ax.plot(fpr, tpr, label=f"{v} (AUC {roc_auc_score(y[mm], sc[mm]):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set(xlabel="Falsos positivos", ylabel="Verdaderos positivos", title="ROC 2025 - mora 30+ a 6 meses")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(SALIDA / "roc_2025.png", dpi=130); plt.close(fig)

    (SALIDA / "registro.json").write_text(json.dumps({"calificado_2025_una_vez": True, "decision": decision},
                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 220)
    for f in ["discriminacion_30_6m", "diferencias_vs_elegida", "monitoreo_confirmacion", "calibracion_por_anio",
              "pruebas_confirmatorias", "equidad_sexo_2025", "efecto_cosecha_con_2025"]:
        print(f"\n== {f}"); print(pd.read_csv(SALIDA / f"{f}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
