"""
07b - Segunda y tercera hipótesis y calibración en el conjunto, sobre el puntaje de la
versión elegida (C3) en la prueba 2025 (diseño metodológico del documento). No reentrena el modelo: usa
resultados/protocolo/reporte_2025/puntajes_2025.parquet y las predicciones fuera de pliegue de 2023-2024.

  H2  Reordenación dentro de la población aprobada: (a) tasa de aprobación constante, (b) deterioro constante,
      frente al segmento vigente (orden aleatorio dentro de cada categoría, en valor esperado: equivale a promediar infinitos sorteos).
      Capa retrospectiva (deterioro y capital observados) y prospectiva (intercepto recalibrado a 10, 13 y 20 %).
  H3  Equidad por sexo y estrato: igualdad de oportunidad (tasa de falsos positivos con "buen pagador" como clase
      favorable) y AUC dentro del grupo, familia confirmatoria con Holm; prueba de proxy; estrato condicionado.
  Calibración en el conjunto: razón observado/esperado e intercepto con la pendiente fija en 1.
  Sensibilidad de la revisión final (exploratoria, --sensibilidad-estado-civil): C3 sin el estado civil, con los
      hiperparámetros de 04a_versiones_cv.py: AUC frente a C3 y a quién deja fuera el corte del 80 %.
Salidas: resultados/hipotesis/h2, h3, calibracion
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import credito as CR
import modelos as M
import protocolo as P
import utilidades_auc as U

warnings.filterwarnings("ignore")
M.NUMERICAS, M.CAT_BAJA, M.CAT_ALTA = P.NUMERICAS, P.CAT_BAJA, P.CAT_ALTA
RAIZ_H = P.RAIZ / "resultados" / "hipotesis"
H2, H3, CAL = RAIZ_H / "h2", RAIZ_H / "h3", RAIZ_H / "calibracion"
COL = P.Y
RIESGO_SEGMENTO = {"BRONCE": 3, "PLATA": 2, "ORO": 1}
N_BOOT = 2000
TASAS_APROBACION = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
METAS_DETERIORO = [0.07, 0.08, 0.09, 0.10]  # 9 %: principal; 7 %: complementario (la mora de la prueba es 9,0 %)
PREVALENCIAS = [0.10, 0.13, 0.20]
PUNTOS = [0.10, 0.20, 0.30]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def grupo_estrato(e):
    return pd.cut(e, [0, 2, 3, 4, 6], labels=["1-2", "3", "4", "5+"]).astype(object)


def cargar():
    decision = json.loads(P.DECISION.read_text(encoding="utf-8"))
    elegida = decision["version"]
    s = pd.read_parquet(P.SALIDA / "reporte_2025" / "puntajes_2025.parquet")
    s = s.merge(CR.variables_credito(s["id_obligacion"].values).reset_index(), on="id_obligacion", how="left",
                validate="one_to_one")
    s["score"] = s[f"score__{elegida}"]
    s["riesgo_segmento"] = s["ref_segmento"].map(RIESGO_SEGMENTO).astype(float)
    s["estrato_g"] = grupo_estrato(s["estrato"])
    return elegida, decision, s


# =============================================================================================== H2
def curva_modelo(y, cap, score):
    """Aprobando de menor a mayor riesgo: deterioro y capital en deterioro acumulados para k = 1..n."""
    o = np.argsort(score, kind="stable")
    yk, ck = np.cumsum(y[o]), np.cumsum(cap[o])
    cdk = np.cumsum((cap * y)[o])
    k = np.arange(1, len(y) + 1)
    return yk / k, cdk / ck


def curva_segmento(y, cap, riesgo):
    """Lo mismo ordenando por segmento con orden aleatorio dentro de cada categoría, en valor esperado: dentro de la categoría que queda en
    el límite, cada crédito entra con la misma probabilidad."""
    n = len(y)
    tasa, capd = np.empty(n), np.empty(n)
    ac_y = ac_c = ac_cd = 0.0
    hecho = 0
    for r in sorted(np.unique(riesgo)):
        m = riesgo == r
        nm, sy, sc, scd = m.sum(), y[m].sum(), cap[m].sum(), (cap * y)[m].sum()
        j = np.arange(1, nm + 1)
        tasa[hecho:hecho + nm] = (ac_y + j * sy / nm) / (hecho + j)
        capd[hecho:hecho + nm] = (ac_cd + j * scd / nm) / (ac_c + j * sc / nm)
        ac_y, ac_c, ac_cd, hecho = ac_y + sy, ac_c + sc, ac_cd + scd, hecho + nm
    return tasa, capd


def resumen_h2(y, cap, score, riesgo):
    tm, cm = curva_modelo(y, cap, score)
    ts, cs = curva_segmento(y, cap, riesgo)
    n = len(y)
    out = {}
    for r in TASAS_APROBACION:
        k = int(np.ceil(r * n)) - 1
        out[f"det_modelo_{r}"], out[f"det_segmento_{r}"] = tm[k], ts[k]
        out[f"cap_modelo_{r}"], out[f"cap_segmento_{r}"] = cm[k], cs[k]
    for t in METAS_DETERIORO:
        km, ks = np.where(tm <= t)[0], np.where(ts <= t)[0]
        out[f"frac_modelo_{t}"] = (km.max() + 1) / n if len(km) else 0.0
        out[f"frac_segmento_{t}"] = (ks.max() + 1) / n if len(ks) else 0.0
    out["det_cartera"], out["cap_cartera"] = y.mean(), (cap * y).sum() / cap.sum()
    return out


def h2(s, poblacion):
    col = f"{P.OBJETIVO}__{poblacion}"
    t = s[s[col].notna()].reset_index(drop=True)
    y, cap = t[col].astype(int).values, t["capital"].values
    score, riesgo = t["score"].values, t["riesgo_segmento"].values
    punto = resumen_h2(y, cap, score, riesgo)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        boots.append(resumen_h2(y[i], cap[i], score[i], riesgo[i]))
    B = pd.DataFrame(boots)
    filas = []
    for r in TASAS_APROBACION:
        for met, nombre in (("det", "deterioro entre aprobados"), ("cap", "capital aprobado en deterioro")):
            d = B[f"{met}_modelo_{r}"] - B[f"{met}_segmento_{r}"]
            filas.append({"poblacion": poblacion, "analisis": "a) aprobación constante", "punto": f"aprobar {r:.0%}",
                          "metrica": nombre, "modelo": punto[f"{met}_modelo_{r}"], "segmento": punto[f"{met}_segmento_{r}"],
                          "cartera_completa": punto[f"{met}_cartera"],
                          "diferencia": punto[f"{met}_modelo_{r}"] - punto[f"{met}_segmento_{r}"],
                          "ic_inf": d.quantile(0.025), "ic_sup": d.quantile(0.975),
                          "p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), "principal": r == 0.80})
    for tt in METAS_DETERIORO:
        d = B[f"frac_modelo_{tt}"] - B[f"frac_segmento_{tt}"]
        filas.append({"poblacion": poblacion, "analisis": "b) deterioro constante", "punto": f"deterioro ≤ {tt:.0%}",
                      "metrica": "fracción aprobable", "modelo": punto[f"frac_modelo_{tt}"],
                      "segmento": punto[f"frac_segmento_{tt}"], "cartera_completa": np.nan,
                      "diferencia": punto[f"frac_modelo_{tt}"] - punto[f"frac_segmento_{tt}"],
                      "ic_inf": d.quantile(0.025), "ic_sup": d.quantile(0.975),
                      "p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), "principal": tt == 0.09})
    # frontera completa (para la figura)
    tm, cm = curva_modelo(y, cap, score)
    ts, cs = curva_segmento(y, cap, riesgo)
    frontera = pd.DataFrame({"poblacion": poblacion, "fraccion_aprobada": np.arange(1, len(y) + 1) / len(y),
                             "det_modelo": tm, "det_segmento": ts, "cap_modelo": cm, "cap_segmento": cs})
    return pd.DataFrame(filas), frontera, t


def h2_prospectiva(t):
    y, cap, ing, score = t[COL].astype(int).values, t["capital"].values, t["ingreso_contratado"].values, t["score"].values
    o = np.argsort(score, kind="stable")
    filas = []
    for pi in PREVALENCIAS:
        a = brentq(lambda a: M.aplicar_intercepto(score, a).mean() - pi, -10, 10)
        p = M.aplicar_intercepto(score, a)
        for r in TASAS_APROBACION + [1.0]:
            k = int(np.ceil(r * len(y)))
            sel = o[:k]
            expo = float((p[sel] * cap[sel]).sum())
            filas.append({"prevalencia_supuesta": pi, "desplazamiento_logit": a, "aprobacion": r,
                          "creditos": k, "capital_aprobado": cap[sel].sum(), "prob_media_aprobados": p[sel].mean(),
                          "exposicion_esperada": expo, "exposicion_sobre_capital": expo / cap[sel].sum(),
                          "ingreso_contratado": ing[sel].sum(), "ingreso_sobre_exposicion": ing[sel].sum() / expo})
    return pd.DataFrame(filas)


def composicion(t):
    """A quién deja fuera cada corte: proporción no aprobada por categoría y perfil de aprobados y no aprobados."""
    t = t.copy()
    t["independiente"] = t["independiente_informal"].map({0: "formal", 1: "informal"})
    t["segmento"] = t["ref_segmento"]
    atributos = ["segmento", "tipo_uso_moto", "independiente", "estado_civil", "nivel_estudio", "sexo", "estrato_g",
                 "zona_homologada"]
    o = np.argsort(t["score"].values, kind="stable")
    filas, resumen = [], []
    for r in (0.70, 0.80, 0.90):
        aprobado = np.zeros(len(t), bool)
        aprobado[o[:int(np.ceil(r * len(t)))]] = True
        t["aprobado"] = aprobado
        y = t[COL].astype(int)
        resumen.append({"aprobacion": r, "aprobados": int(aprobado.sum()), "no_aprobados": int((~aprobado).sum()),
                        "deterioro_aprobados": y[aprobado].mean(), "deterioro_no_aprobados": y[~aprobado].mean(),
                        "eventos_no_aprobados": int(y[~aprobado].sum()), "pct_eventos_evitados": y[~aprobado].sum() / y.sum(),
                        "capital_medio_aprobados": t.loc[aprobado, "capital"].mean(),
                        "capital_medio_no_aprobados": t.loc[~aprobado, "capital"].mean(),
                        "pct_capital_no_aprobado": t.loc[~aprobado, "capital"].sum() / t["capital"].sum()})
        for a in atributos:
            g = t.groupby(t[a].astype(str), dropna=False)
            tab = pd.DataFrame({"creditos": g.size(), "pct_cartera": g.size() / len(t),
                                "pct_entre_no_aprobados": g["aprobado"].apply(lambda s: (~s).sum()) / (~aprobado).sum(),
                                "tasa_no_aprobacion": g["aprobado"].apply(lambda s: (~s).mean()),
                                "deterioro_observado": g[COL].mean()})
            filas.append(tab.reset_index().rename(columns={a: "categoria"}).assign(atributo=a, aprobacion=r))
    comp = pd.concat(filas)[["aprobacion", "atributo", "categoria", "creditos", "pct_cartera", "pct_entre_no_aprobados",
                             "tasa_no_aprobacion", "deterioro_observado"]]
    comp.to_csv(H2 / "composicion.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(resumen).to_csv(H2 / "composicion_resumen.csv", index=False, encoding="utf-8-sig")


# =============================================================================================== sensibilidad
SENS = "S_C3_sin_estado_civil"  # protocolo.SENSIBILIDAD
EDADES = ([17, 25, 30, 35, 40, 50, 90], ["18-25", "26-30", "31-35", "36-40", "41-50", "51+"])  # como en 08_metricas_interpretabilidad.py


def sensibilidad_estado_civil():
    """C3 sin el estado civil, ajustada con el mismo procedimiento de las versiones (04a_versiones_cv.py): AUC frente a C3
    en la CV 5x5 (Nadeau-Bengio) y en 2025 (DeLong), y tasa de no aprobación al aprobar el 80 % con cada puntaje, por
    estado civil y por edad. Exploratoria: no cambia la decisión."""
    elegida, _, s = cargar()
    variables = P.SENSIBILIDAD[SENS]
    params = json.loads((P.SALIDA / "parametros.json").read_text(encoding="utf-8"))
    base = P.cargar_reporte()
    dev, oot = base[base.periodo == "train"], base[base.periodo == "oot"].copy()
    d = dev[dev[COL].notna()]
    modelos = [M.construir(n, params[f"{SENS}|{n}"], variables).fit(d, d[COL].astype(int).values) for n in M.N_TRIALS]
    oot["score_sin_ec"] = np.mean([m.predict_proba(oot)[:, 1] for m in modelos], axis=0)
    t = s.merge(oot[["id_obligacion", "score_sin_ec"]], on="id_obligacion", how="left", validate="one_to_one")
    t = t[t[COL].notna()].reset_index(drop=True)  # la misma muestra y el mismo orden de la composición de H2
    y = t[COL].astype(int).values
    a1, a2, dif, _, i1, i2, p = U.delong(y, t["score_sin_ec"].values, t["score"].values)
    oof = lambda v: pd.read_parquet(P.SALIDA / "oof" / f"{v}__{P.ETIQUETA}.parquet").query("modelo == 'ensamble'")
    A = pd.DataFrame({v: oof(v).groupby(["semilla", "pliegue"]).apply(lambda g: P.auc_intra(g.y, g.p, g.anio))
                      for v in (SENS, elegida)})
    m_cv, _, c1, c2, p_cv = U.nadeau_bengio((A[SENS] - A[elegida]).values)
    pd.DataFrame([{"version": SENS, "variables": ", ".join(variables), "auc_cv": A[SENS].mean(),
                   "auc_cv_C3": A[elegida].mean(), "dif_cv": m_cv, "cv_ic_inf": c1, "cv_ic_sup": c2, "p_cv": p_cv,
                   "auc_2025": a1, "auc_2025_C3": a2, "dif_2025": dif, "ic_inf_2025": i1, "ic_sup_2025": i2,
                   "p_2025": p}]).to_csv(H2 / "sensibilidad_estado_civil_auc.csv", index=False, encoding="utf-8-sig")
    t["edad_g"] = pd.cut(t["edad"], EDADES[0], labels=EDADES[1]).astype(str)
    filas, resumen = [], []
    for nombre, col in (("C3", "score"), ("sin estado civil", "score_sin_ec")):
        o = np.argsort(t[col].values, kind="stable")
        aprobado = np.zeros(len(t), bool)
        aprobado[o[:int(np.ceil(0.80 * len(t)))]] = True
        resumen.append({"puntaje": nombre, "aprobados": int(aprobado.sum()), "deterioro_aprobados": y[aprobado].mean(),
                        "deterioro_no_aprobados": y[~aprobado].mean()})
        for a in ("estado_civil", "edad_g"):
            g = pd.Series(~aprobado).groupby(t[a].astype(str).values)
            filas.append(pd.DataFrame({"atributo": a, "categoria": g.size().index, "creditos": g.size().values,
                                       "puntaje": nombre, "tasa_no_aprobacion": g.mean().values}))
    comp = (pd.concat(filas).pivot_table(index=["atributo", "categoria", "creditos"], columns="puntaje",
                                         values="tasa_no_aprobacion").reset_index()
            [["atributo", "categoria", "creditos", "C3", "sin estado civil"]])
    comp.to_csv(H2 / "sensibilidad_estado_civil_composicion.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(resumen).to_csv(H2 / "sensibilidad_estado_civil_resumen.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for f in ["auc", "resumen", "composicion"]:
        print(f"\n== sensibilidad_estado_civil_{f}")
        print(pd.read_csv(H2 / f"sensibilidad_estado_civil_{f}.csv", encoding="utf-8-sig").round(4).to_string(index=False))


# =============================================================================================== H3
def metricas_grupo(y, sc, umbral):
    marc = sc >= umbral
    n_ev, n_ne = int(y.sum()), int((y == 0).sum())
    return {"n": len(y), "eventos": n_ev, "tasa_observada": y.mean(), "prob_media": sc.mean(),
            "observado_sobre_esperado": y.mean() / sc.mean(), "pct_marcados": marc.mean(),
            "tpr": (marc & (y == 1)).sum() / n_ev if n_ev else np.nan,
            "fpr": (marc & (y == 0)).sum() / n_ne if n_ne else np.nan,
            "auc": roc_auc_score(y, sc) if 0 < n_ev < len(y) else np.nan}


def tabla_grupos(t, y, sc, atributo, f):
    umbral = np.quantile(sc, 1 - f)
    filas = []
    for g in sorted(t[atributo].dropna().unique(), key=str):
        k = (t[atributo] == g).values
        m = metricas_grupo(y[k], sc[k], umbral)
        filas.append({"atributo": atributo, "grupo": g, "punto": f, **m, "interpretable": m["n"] >= 30 and m["eventos"] >= 10})
    return pd.DataFrame(filas)


def brechas(t, y, sc, atributo, ref, grupos, f):
    """Brecha grupo - referencia en FPR, AUC, % marcados, TPR y O/E; bootstrap estratificado por grupo."""
    rng = np.random.default_rng(1)
    idx = {g: np.where((t[atributo] == g).values)[0] for g in grupos + [ref]}
    resto = np.where(~t[atributo].isin(grupos + [ref]).values)[0]
    metr = ["fpr", "auc", "pct_marcados", "tpr", "observado_sobre_esperado"]

    def calcular(ind):
        yy, ss = y[ind], sc[ind]
        umbral = np.quantile(ss, 1 - f)
        pos = {}
        inicio = 0
        for g in grupos + [ref]:
            pos[g] = np.arange(inicio, inicio + len(idx_b[g]))
            inicio += len(idx_b[g])
        mg = {g: metricas_grupo(yy[pos[g]], ss[pos[g]], umbral) for g in grupos + [ref]}
        return {(g, m): mg[g][m] - mg[ref][m] for g in grupos for m in metr}

    idx_b = idx
    orden = np.concatenate([idx[g] for g in grupos + [ref]] + [resto])
    punto = calcular(orden)
    boots = []
    for _ in range(N_BOOT):
        idx_b = {g: rng.choice(v, len(v)) for g, v in idx.items()}
        r = rng.choice(resto, len(resto)) if len(resto) else resto
        boots.append(calcular(np.concatenate([idx_b[g] for g in grupos + [ref]] + [r])))
    B = pd.DataFrame(boots)
    filas = []
    for g in grupos:
        for m in metr:
            d = B[(g, m)].dropna()
            filas.append({"atributo": atributo, "grupo": g, "referencia": ref, "punto": f, "metrica": m,
                          "brecha": punto[(g, m)], "ic_inf": d.quantile(0.025), "ic_sup": d.quantile(0.975),
                          "p": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                          "confirmatoria": f == 0.20 and m in ("fpr", "auc")})
    return pd.DataFrame(filas)


def h3(s, elegida):
    t = s[s[COL].notna()].reset_index(drop=True)
    y, sc = t[COL].astype(int).values, t["score"].values
    grupos = pd.concat([tabla_grupos(t, y, sc, a, f) for a in ("sexo", "estrato_g") for f in PUNTOS])
    grupos.to_csv(H3 / "grupos_2025.csv", index=False, encoding="utf-8-sig")
    g20 = grupos[grupos.punto == 0.20]
    ref_estrato = g20[g20.atributo == "estrato_g"].sort_values("n").iloc[-1]["grupo"]
    interpretables = set(g20.loc[g20.interpretable, "grupo"])
    fam = []
    for f in PUNTOS:
        fam.append(brechas(t, y, sc, "sexo", "M", ["F"], f))
        otros = [g for g in ["1-2", "3", "4", "5+"] if g != ref_estrato]
        fam.append(brechas(t, y, sc, "estrato_g", ref_estrato, otros, f))
    br = pd.concat(fam, ignore_index=True)
    br["en_familia"] = br.confirmatoria & br.grupo.isin(interpretables) & (br.referencia.isin(interpretables))
    br["p_holm"] = np.nan
    br.loc[br.en_familia, "p_holm"] = U.holm(br.loc[br.en_familia, "p"].values)
    br.to_csv(H3 / "brechas_2025.csv", index=False, encoding="utf-8-sig")

    # complemento: predicciones fuera de pliegue 2023-2024
    oof = pd.read_parquet(P.SALIDA / "oof" / f"{elegida}__{P.ETIQUETA}.parquet")
    oof = oof[oof.modelo == "ensamble"].groupby(["id_obligacion", "anio", "y"], as_index=False)["p"].mean()
    base = pd.read_parquet(P.RAIZ / "datos" / "base_protocolo.parquet")[["id_obligacion", "sexo", "estrato"]]
    oof = oof.merge(base, on="id_obligacion")
    oof["estrato_g"] = grupo_estrato(oof["estrato"])
    pd.concat([tabla_grupos(oof, oof.y.values, oof.p.values, a, 0.20) for a in ("sexo", "estrato_g")]) \
        .to_csv(H3 / "grupos_oof_2023_2024.csv", index=False, encoding="utf-8-sig")

    # prueba de proxy: ¿las variables de C3 o el puntaje reconstruyen el sexo o el estrato?
    pob = P.cargar_reporte()
    pob = pob[pob[COL].notna()].reset_index(drop=True)
    pob["estrato_g"] = grupo_estrato(pob["estrato"])
    variables = json.loads(P.DECISION.read_text(encoding="utf-8"))["variables"]
    proxy = []
    objetivos = {"sexo = M": pob["sexo"] == "M", "estrato 5+": pob["estrato_g"] == "5+", "estrato 1-2": pob["estrato_g"] == "1-2"}
    for nombre, obj in objetivos.items():
        k = pob["estrato_g"].notna().values if nombre.startswith("estrato") else np.ones(len(pob), bool)
        d, yy = pob[k], obj[k].astype(int).values
        cvp = cross_val_predict(M.construir("logistica", {"C": 1.0}, variables), d, yy,
                                cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        kt = t["estrato_g"].notna().values if nombre.startswith("estrato") else np.ones(len(t), bool)
        yt = (t["sexo"] == "M").values if nombre == "sexo = M" else (t["estrato_g"] == nombre.split()[1]).values
        proxy.append({"atributo": nombre, "n": int(k.sum()), "prevalencia": yy.mean(),
                      "auc_desde_variables_C3_cv": roc_auc_score(yy, cvp),
                      "auc_desde_puntaje_2025": roc_auc_score(yt[kt].astype(int), sc[kt])})
    pd.DataFrame(proxy).to_csv(H3 / "prueba_proxy.csv", index=False, encoding="utf-8-sig")

    # estrato condicionado al producto y a las condiciones del crédito (descriptivo, población con resultado)
    pob = pob.merge(CR.variables_credito(pob["id_obligacion"].values).reset_index(), on="id_obligacion", how="left")
    pob = pob[pob["estrato_g"].notna()].copy()
    pob["y"] = pob[COL].astype(int)
    vc = pob["familia_moto"].value_counts()
    pob["familia"] = pob["familia_moto"].where(pob["familia_moto"].map(vc) >= 30, "OTRA")
    pob["log_precio"], pob["log_capital"] = np.log(pob["precio_moto"]), np.log(pob["capital"])
    tasas = pob.groupby(["estrato_g", "anio"])["y"].agg(["size", "mean"]).unstack()
    tasas.to_csv(H3 / "estrato_tasas_por_anio.csv", encoding="utf-8-sig")
    ref = "C(estrato_g, Treatment('4'))"
    formulas = {"estrato + año": f"y ~ {ref} + C(anio)",
                "+ producto y condiciones": f"y ~ {ref} + C(anio) + C(familia) + log_precio + log_capital + plazo "
                                            f"+ cuota_inicial_pct + prenda"}
    filas = []
    for nombre, fml in formulas.items():
        fit = smf.logit(fml, pob).fit(disp=0)
        ci = fit.conf_int()
        for term in [c for c in fit.params.index if c.startswith("C(estrato_g")]:
            filas.append({"modelo": nombre, "termino": term.split("[T.")[1].rstrip("]") + " frente a 4",
                          "odds_ratio": np.exp(fit.params[term]), "ic_inf": np.exp(ci.loc[term, 0]),
                          "ic_sup": np.exp(ci.loc[term, 1]), "p": fit.pvalues[term], "n": int(fit.nobs)})
    pd.DataFrame(filas).to_csv(H3 / "estrato_condicionado.csv", index=False, encoding="utf-8-sig")
    return ref_estrato


# =============================================================================================== calibración
def cal_conjunto(y, p):
    y, p = np.asarray(y), np.asarray(p)
    fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=logit(p)).fit()
    ci = fit.conf_int()[0]
    return {"n": len(y), "tasa_observada": y.mean(), "prob_media": p.mean(), "observado_sobre_esperado": y.mean() / p.mean(),
            "intercepto_offset": float(fit.params[0]), "ic_inf": float(ci[0]), "ic_sup": float(ci[1])}


def calibracion(s, elegida):
    t = s[s[COL].notna()].reset_index(drop=True)
    y, sc = t[COL].astype(int).values, t["score"].values
    t1 = (t.fecha_desembolso < P.CORTE_RECALIBRACION).values
    a = M.recalibrar_intercepto(y[t1], sc[t1])
    oof = pd.read_parquet(P.SALIDA / "oof" / f"{elegida}__{P.ETIQUETA}.parquet")
    oof = oof[oof.modelo == "ensamble"].groupby(["id_obligacion", "anio", "y"], as_index=False)["p"].mean()
    filas = [{"periodo": str(an), "fuente": "CV (fuera de pliegue)", **cal_conjunto(g.y, g.p)} for an, g in oof.groupby("anio")]
    filas += [{"periodo": "2025", "fuente": "prueba", **cal_conjunto(y, sc)},
              {"periodo": "2025-T2", "fuente": "prueba sin recalibrar", **cal_conjunto(y[~t1], sc[~t1])},
              {"periodo": "2025-T2", "fuente": "prueba recalibrada con 2025-T1", **cal_conjunto(y[~t1], M.aplicar_intercepto(sc[~t1], a))}]
    for atr in ("sexo", "estrato_g"):
        for g in sorted(t[atr].dropna().unique(), key=str):
            k = (t[atr] == g).values
            filas.append({"periodo": "2025", "fuente": f"prueba · {atr} {g}", **cal_conjunto(y[k], sc[k])})
    pd.DataFrame(filas).to_csv(CAL / "calibracion_en_el_conjunto.csv", index=False, encoding="utf-8-sig")


def figuras():
    """Figuras de H2 (frontera.png) y H3 (brechas.png), generadas al ancho con que se imprimen en el documento (5,71
    pulgadas, 14,5 cm), para que todo el texto mida al menos 9 pt al tamaño final; ejes con coma decimal."""
    import re

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    class FormatoComa(ScalarFormatter):
        def __call__(self, x, pos=None):
            s = super().__call__(x, pos).replace(".", ",")
            return re.sub(r"(\d)(?=(\d{3})+(?!\d))", r"\1.", s) if re.fullmatch(r"[−-]?\d{4,}", s) else s

    estilo = {"font.family": "Arial", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 9,
              "ytick.labelsize": 9, "legend.fontsize": 9}
    fr = pd.read_csv(H2 / "frontera.csv")
    fr = fr[fr.poblacion == P.ETIQUETA]
    n_fr = len(fr)
    fr = fr[fr.fraccion_aprobada >= 0.5]
    with plt.rc_context(estilo):
        fig, ax = plt.subplots(1, 2, figsize=(5.71, 2.18))
        for k, (met, titulo) in enumerate([("det", "Deterioro entre los aprobados"), ("cap", "Capital aprobado en deterioro")]):
            ax[k].plot(fr.fraccion_aprobada * 100, fr[f"{met}_modelo"] * 100, label="Ordenando por el puntaje (C3)")
            ax[k].plot(fr.fraccion_aprobada * 100, fr[f"{met}_segmento"] * 100, label="Ordenando por el segmento vigente")
            ax[k].set(xlabel="Proporción aprobada (%)", ylabel="%", title=titulo)
            ax[k].grid(alpha=0.3)
            ax[k].xaxis.set_major_formatter(FormatoComa())
            ax[k].yaxis.set_major_formatter(FormatoComa())
        fig.suptitle(f"Reordenación dentro de la población aprobada - prueba 2025 ({n_fr:,} créditos)".replace(",", "."),
                     fontsize=9.5)
        fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.5)
        fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, handlelength=1.5)
        fig.savefig(H2 / "frontera.png", dpi=300)
        plt.close(fig)

    br = pd.read_csv(H3 / "brechas_2025.csv")
    br = br[(br.punto == 0.20) & br.metrica.isin(["fpr", "auc"])].copy()
    br["etiqueta"] = np.where(br.atributo == "sexo", "Mujeres frente a hombres", "Estrato " + br.grupo + " frente a " + br.referencia)
    with plt.rc_context(estilo):
        fig, ax = plt.subplots(1, 2, figsize=(5.71, 1.87), sharey=True)
        for k, (met, titulo) in enumerate([("fpr", "Brecha en la tasa de\nfalsos positivos"),
                                           ("auc", "Brecha en el AUC\ndentro del grupo")]):
            g = br[br.metrica == met].reset_index(drop=True)
            ax[k].errorbar(g.brecha, range(len(g)), xerr=[g.brecha - g.ic_inf, g.ic_sup - g.brecha], fmt="o", ms=4, capsize=3)
            ax[k].axvline(0, color="grey", lw=0.8)
            ax[k].set_yticks(range(len(g)), g.etiqueta)
            ax[k].set(title=titulo, xlabel="Diferencia (IC 95 %)")
            ax[k].xaxis.set_major_formatter(FormatoComa())
        fig.suptitle("Equidad de C3 en 2025 al marcar el 20 % de mayor riesgo", fontsize=9.5)
        fig.tight_layout(w_pad=1.5)
        fig.savefig(H3 / "brechas.png", dpi=300)
        plt.close(fig)


def main():
    for d in (H2, H3, CAL):
        d.mkdir(parents=True, exist_ok=True)
    if "--solo-figuras" in __import__("sys").argv:
        figuras()
        return
    if "--sensibilidad-estado-civil" in __import__("sys").argv:
        sensibilidad_estado_civil()
        return
    if "--solo-composicion" in __import__("sys").argv:
        _, _, s = cargar()
        composicion(s[s[COL].notna()].reset_index(drop=True))
        return
    elegida, decision, s = cargar()
    res, fronteras = [], []
    r, fr, t = h2(s, P.ETIQUETA)
    res.append(r); fronteras.append(fr)
    h2_prospectiva(t).to_csv(H2 / "prospectiva_prevalencias.csv", index=False, encoding="utf-8-sig")
    composicion(t)
    pd.concat(res).to_csv(H2 / "reordenacion.csv", index=False, encoding="utf-8-sig")
    pd.concat(fronteras).to_csv(H2 / "frontera.csv", index=False, encoding="utf-8-sig")
    print("H2 listo", flush=True)
    ref = h3(s, elegida)
    print("H3 listo; referencia de estrato:", ref, flush=True)
    calibracion(s, elegida)
    figuras()
    print("calibración y figuras listas", flush=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for f in ["h2/reordenacion", "h2/prospectiva_prevalencias", "h3/grupos_2025", "h3/brechas_2025", "h3/prueba_proxy",
              "h3/estrato_condicionado", "h3/estrato_tasas_por_anio", "calibracion/calibracion_en_el_conjunto"]:
        print(f"\n== {f}"); print(pd.read_csv(RAIZ_H / f"{f}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
