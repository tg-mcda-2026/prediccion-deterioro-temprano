"""
08 - Métricas completas, algoritmos por separado e interpretabilidad de la versión elegida (C3),
con el mismo ajuste del reporte de 2025 (todo 2023-2024, hiperparámetros de 04a_versiones_cv.py, etiqueta principal). No cambia la decisión.

  1. Métricas en 2025 del ensamble, de cada algoritmo y del segmento: AUC (IC de DeLong), Gini, KS, PR-AUC (IC por
     bootstrap), Brier y calibración; estabilidad de cada algoritmo en la validación cruzada 2023-2024.
  2. Métricas a un umbral (marcar el 10, 20 y 30 % de mayor riesgo): sensibilidad, especificidad, precisión, F1 y
     deterioro entre los no marcados, con IC de Wilson.
  3. Interpretabilidad del ensamble: valores de Shapley exactos (6 variables, 64 coaliciones, fondo de 100 créditos
     de 2023-2024) sobre la prueba 2025, importancia global, dependencia por categoría y explicación de tres créditos;
     importancia por permutación (caída del AUC al permutar cada variable).
Salidas: resultados/protocolo/c3_detalle/
"""
import json
import re
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

import limpieza as L
import modelos as M
import protocolo as P
import utilidades_auc as U

warnings.filterwarnings("ignore")
M.NUMERICAS, M.CAT_BAJA, M.CAT_ALTA = P.NUMERICAS, P.CAT_BAJA, P.CAT_ALTA
MODELOS = list(M.N_TRIALS)
SALIDA = P.SALIDA / "c3_detalle"
COL = P.Y
RIESGO_SEGMENTO = {"BRONCE": 3, "PLATA": 2, "ORO": 1}
NOMBRE_VAR = {"edad": "Edad", "estado_civil": "Estado civil", "ciudad": "Ciudad", "independiente_informal": "Independiente informal",
              "nivel_estudio": "Nivel de estudio", "familia_moto": "Familia de moto"}


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def pr_auc_ic(y, p, n=1000, semilla=0):
    rng = np.random.default_rng(semilla)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [average_precision_score(y[i], p[i]) for i in
         (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    fit = smf.logit("y ~ lp", data=pd.DataFrame({"y": y, "lp": lp})).fit(disp=0)
    return float(fit.params["lp"]), float(fit.params["Intercept"])


def fila_metricas(nombre, y, p, probabilidad=True):
    auc, i1, i2 = U.ic_auc_delong(y, p)
    pr1, pr2 = pr_auc_ic(y, p)
    f = {"modelo": nombre, "auc": auc, "auc_ic_inf": i1, "auc_ic_sup": i2, "gini": 2 * auc - 1, "ks": ks(y, p),
         "pr_auc": average_precision_score(y, p), "pr_auc_ic_inf": pr1, "pr_auc_ic_sup": pr2, "prevalencia": y.mean()}
    if probabilidad:
        pend, inter = calib(y, p)
        f.update({"brier": brier_score_loss(y, p), "prob_media": p.mean(), "pendiente_cal": pend, "intercepto_cal": inter})
    return f


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    decision = json.loads(P.DECISION.read_text(encoding="utf-8"))
    elegida, variables = decision["version"], decision["variables"]
    params = json.loads((P.SALIDA / "parametros.json").read_text(encoding="utf-8"))
    base = P.cargar_reporte()
    dev, oot = base[base.periodo == "train"], base[base.periodo == "oot"]
    d = dev[dev[COL].notna()].reset_index(drop=True)
    yd = d[COL].astype(int).values
    t = oot[oot[COL].notna()].reset_index(drop=True)
    y = t[COL].astype(int).values

    modelos = {n: M.construir(n, params[f"{elegida}|{n}"], variables).fit(d, yd) for n in MODELOS}
    ensamble = lambda X: np.mean([m.predict_proba(X)[:, 1] for m in modelos.values()], axis=0)
    p_ens = ensamble(t)
    ref = pd.read_parquet(P.SALIDA / "reporte_2025" / "puntajes_2025.parquet").set_index("id_obligacion")
    assert np.allclose(p_ens, ref.loc[t["id_obligacion"], f"score__{elegida}"].values, atol=1e-9), "no reproduce el puntaje"
    print("ensamble reproducido", flush=True)

    # ---------------------------------------------------------------- 1. métricas en 2025 y estabilidad en CV
    oof = pd.read_parquet(P.SALIDA / "oof" / f"{elegida}__{P.ETIQUETA}.parquet")
    filas, difs = [], []
    for n in MODELOS + ["ensamble"]:
        p = p_ens if n == "ensamble" else modelos[n].predict_proba(t)[:, 1]
        f = fila_metricas(n, y, p)
        cv = oof[oof.modelo == n].groupby(["semilla", "pliegue"]).apply(lambda g: P.auc_intra(g.y, g.p, g.anio))
        f.update({"auc_cv_intra_cosecha": cv.mean(), "auc_cv_de_pliegues": cv.std()})
        filas.append(f)
        if n != "ensamble":
            _, _, dd, _, a, b, pv = U.delong(y, p, p_ens)
            difs.append({"modelo": n, "frente_a": "ensamble", "diferencia_auc_2025": dd, "ic_inf": a, "ic_sup": b, "p": pv})
    seg = t["ref_segmento"].map(RIESGO_SEGMENTO).astype(float).values
    filas.append(fila_metricas("segmento vigente", y, seg, probabilidad=False))
    pd.DataFrame(filas).to_csv(SALIDA / "metricas_2025.csv", index=False, encoding="utf-8-sig")
    dd = pd.DataFrame(difs)
    dd["p_holm"] = U.holm(dd.p.values)
    dd.to_csv(SALIDA / "algoritmos_vs_ensamble_2025.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------------- 2. métricas a un umbral
    um = []
    for frac in (0.10, 0.20, 0.30):
        marc = p_ens >= np.quantile(p_ens, 1 - frac)
        vp, fp = int((marc & (y == 1)).sum()), int((marc & (y == 0)).sum())
        fn, vn = int((~marc & (y == 1)).sum()), int((~marc & (y == 0)).sum())
        sens, esp, prec = vp / (vp + fn), vn / (vn + fp), vp / (vp + fp)
        um.append({"marcados": f"{frac:.0%}", "n_marcados": int(marc.sum()), "vp": vp, "fp": fp, "fn": fn, "vn": vn,
                   "sensibilidad": sens, "sens_ic_inf": U.wilson(vp, vp + fn)[0], "sens_ic_sup": U.wilson(vp, vp + fn)[1],
                   "especificidad": esp, "esp_ic_inf": U.wilson(vn, vn + fp)[0], "esp_ic_sup": U.wilson(vn, vn + fp)[1],
                   "precision": prec, "prec_ic_inf": U.wilson(vp, vp + fp)[0], "prec_ic_sup": U.wilson(vp, vp + fp)[1],
                   "f1": 2 * prec * sens / (prec + sens), "deterioro_no_marcados": fn / (fn + vn),
                   "prevalencia": y.mean()})
    pd.DataFrame(um).to_csv(SALIDA / "metricas_umbral_2025.csv", index=False, encoding="utf-8-sig")
    print("métricas listas", flush=True)

    # ---------------------------------------------------------------- 3. interpretabilidad: Shapley exactos
    # el enmascarador de shap trabaja con números: las categorías se codifican como enteros y se decodifican antes de
    # pasar los datos al modelo, que recibe exactamente las mismas columnas que en el ajuste
    tipos = d[variables].dtypes
    cats = {v: np.array(sorted(set(d[v]) | set(t[v])), dtype=object) for v in variables if tipos[v] == object}

    def codificar(df):
        out = df[variables].copy()
        for v, c in cats.items():
            out[v] = out[v].map({k: i for i, k in enumerate(c)})
        return out.astype(float).values

    def f_shap(X):
        df = pd.DataFrame(X, columns=variables)
        for v, c in cats.items():
            df[v] = c[np.rint(df[v].values).astype(int)]
        return ensamble(df.astype(tipos))

    fondo = codificar(d.sample(100, random_state=0))
    assert np.allclose(f_shap(codificar(t)), p_ens), "la codificación para shap altera las predicciones"
    t0 = time.time()
    explicador = shap.explainers.Exact(f_shap, shap.maskers.Independent(fondo, max_samples=100))
    sv = explicador(codificar(t))
    print(f"Shapley calculados en {time.time() - t0:.0f}s", flush=True)
    valores = pd.DataFrame(sv.values, columns=variables)
    base_valor = float(np.mean(sv.base_values))
    assert np.allclose(valores.sum(axis=1) + sv.base_values, p_ens, atol=1e-6), "los Shapley no suman la predicción"
    glob = pd.DataFrame({"variable": variables, "media_abs_shap": valores.abs().mean().values,
                         "media_shap": valores.mean().values}).sort_values("media_abs_shap", ascending=False)
    glob["pct_del_total"] = glob.media_abs_shap / glob.media_abs_shap.sum()
    glob.to_csv(SALIDA / "shap_global.csv", index=False, encoding="utf-8-sig")
    largo = valores.assign(id_obligacion=t["id_obligacion"].values).melt(id_vars="id_obligacion", var_name="variable",
                                                                          value_name="shap")
    largo["valor"] = t[variables].astype(str).melt(value_name="valor")["valor"].values
    largo.to_parquet(SALIDA / "shap_valores_2025.parquet", index=False)
    # dependencia: contribución media por categoría (o tramo de edad) y su frecuencia
    dep = []
    for v in variables:
        x = t[v]
        if v == "edad":
            x = pd.cut(x, [17, 25, 30, 35, 40, 50, 90], labels=["18-25", "26-30", "31-35", "36-40", "41-50", "51+"]).astype(str)
        elif v == "ciudad":
            top = x.value_counts().index[:8]
            x = x.where(x.isin(top), "OTRAS")
        g = pd.DataFrame({"categoria": x.astype(str).values, "shap": valores[v].values, "y": y}).groupby("categoria")
        dep.append(g.agg(creditos=("shap", "size"), shap_medio=("shap", "mean"), deterioro_observado=("y", "mean"))
                   .reset_index().assign(variable=v))
    pd.concat(dep).sort_values(["variable", "shap_medio"]).to_csv(SALIDA / "shap_por_categoria.csv", index=False,
                                                                   encoding="utf-8-sig")
    # explicación local de tres créditos
    umbral20 = np.quantile(p_ens, 0.80)
    casos = {"mayor riesgo": int(np.argmax(p_ens)), "junto al corte del 20 %": int(np.argmin(np.abs(p_ens - umbral20))),
             "menor riesgo": int(np.argmin(p_ens))}
    loc = []
    for nombre, i in casos.items():
        for v in variables:
            loc.append({"caso": nombre, "probabilidad": p_ens[i], "valor_base": float(sv.base_values[i]),
                        "variable": v, "valor": str(t.loc[i, v]), "contribucion": valores.loc[i, v]})
    pd.DataFrame(loc).to_csv(SALIDA / "shap_local.csv", index=False, encoding="utf-8-sig")

    figura_global(glob, valores)

    figura_local(pd.DataFrame(loc))

    # importancia por permutación (caída del AUC en 2025)
    rng = np.random.default_rng(1)
    auc0 = roc_auc_score(y, p_ens)
    perm = []
    for v in variables:
        caidas = []
        for _ in range(20):
            X = t[variables].copy()
            X[v] = rng.permutation(X[v].values)
            caidas.append(auc0 - roc_auc_score(y, ensamble(X)))
        perm.append({"variable": v, "caida_auc_media": np.mean(caidas), "caida_auc_de": np.std(caidas, ddof=1)})
    pd.DataFrame(perm).sort_values("caida_auc_media", ascending=False).to_csv(SALIDA / "importancia_permutacion_2025.csv",
                                                                               index=False, encoding="utf-8-sig")
    print("interpretabilidad lista", flush=True)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
    for f in ["metricas_2025", "algoritmos_vs_ensamble_2025", "metricas_umbral_2025", "shap_global",
              "importancia_permutacion_2025", "shap_local"]:
        print(f"\n== {f}"); print(pd.read_csv(SALIDA / f"{f}.csv").round(4).to_string(index=False))


class FormatoComa(ScalarFormatter):
    """Rótulos numéricos de los ejes con coma decimal y punto de miles, como en el texto del documento."""

    def __call__(self, x, pos=None):
        s = super().__call__(x, pos).replace(".", ",")
        return re.sub(r"(\d)(?=(\d{3})+(?!\d))", r"\1.", s) if re.fullmatch(r"[−-]?\d{4,}", s) else s


ESTILO_DOCUMENTO = {"font.family": "Arial", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9}


def figura_global(glob, valores):
    """Importancia global (media del valor absoluto de Shapley) y contribución de cada crédito de la prueba 2025. Se
    genera al ancho con que se imprime en el documento (5,71 pulgadas, 14,5 cm), para que todo el texto mida al menos
    9 pt al tamaño final; los dos paneles comparten el eje de las variables."""
    g2 = glob.iloc[::-1]
    orden = list(g2.variable)
    with plt.rc_context(ESTILO_DOCUMENTO):
        fig, ax = plt.subplots(1, 2, figsize=(5.71, 2.0), sharey=True)
        ax[0].barh(range(len(orden)), g2.media_abs_shap * 100)
        ax[0].set_yticks(range(len(orden)), [NOMBRE_VAR[v] for v in orden])
        ax[0].set(xlabel="Contribución media absoluta\n(puntos porcentuales\nde probabilidad)", title="Importancia global\n(Shapley)")
        rng = np.random.default_rng(0)
        for k, v in enumerate(orden):
            ax[1].scatter(valores[v] * 100, k + rng.uniform(-0.25, 0.25, len(valores)), s=3, alpha=0.4)
        ax[1].axvline(0, color="grey", lw=0.8)
        ax[1].set(xlabel="Contribución a la probabilidad\n(puntos porcentuales)", title="Contribución por crédito\n(prueba 2025)")
        for a in ax:
            a.xaxis.set_major_formatter(FormatoComa())
        fig.tight_layout(w_pad=1.0)
        fig.savefig(SALIDA / "shap_global.png", dpi=300)
        plt.close(fig)


def figura_local(lc):
    """Figura de los tres créditos con la familia de moto en su etiqueta genérica (limpieza.etiquetas_familia).
    Se genera al ancho con que se imprime en el documento (5,71 pulgadas, 14,5 cm), para que todo el texto mida al menos
    9 pt al tamaño final: los tres paneles van apilados, con el eje x compartido, y las cifras con coma decimal."""
    b = pd.read_parquet(P.RAIZ / "datos" / "base_protocolo.parquet", columns=["familia_moto", "tipo_uso_moto"])
    etiqueta = L.etiquetas_familia(b)
    pct = lambda x: f"{x * 100:.1f}".replace(".", ",") + "\u00a0%"
    estilo = {"font.family": "Arial", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 9,
              "ytick.labelsize": 9}
    with plt.rc_context(estilo):
        fig, ax = plt.subplots(3, 1, figsize=(5.71, 4.15), sharex=True)
        for k, nombre in enumerate(lc.caso.unique()):
            c = lc[lc.caso == nombre].sort_values("contribucion")
            mostrados = [etiqueta.get(val, val) if v == "familia_moto" else re.sub(r"^(\d+)\.0$", r"\1", str(val))
                         for v, val in zip(c.variable, c.valor)]
            ax[k].barh([f"{NOMBRE_VAR[v]} = {str(val)[:24]}" for v, val in zip(c.variable, mostrados)],
                       c.contribucion * 100, height=0.7, color=["#2a9d8f" if x < 0 else "#9b2226" for x in c.contribucion])
            ax[k].axvline(0, color="grey", lw=0.8)
            ax[k].set_title(f"{nombre.capitalize()}: {pct(c.probabilidad.iloc[0])} (base {pct(c.valor_base.iloc[0])})",
                            loc="left")
            ax[k].grid(axis="x", alpha=0.3)
        ax[-1].set_xlabel("Contribución (puntos porcentuales)")
        fig.tight_layout(h_pad=0.6)
        fig.savefig(SALIDA / "shap_local.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    if "--figuras" in sys.argv:  # solo las figuras, desde shap_global.csv, shap_valores_2025.parquet y shap_local.csv
        largo = pd.read_parquet(SALIDA / "shap_valores_2025.parquet")
        valores = pd.DataFrame({v: largo.loc[largo.variable == v, "shap"].values for v in largo.variable.unique()})
        figura_global(pd.read_csv(SALIDA / "shap_global.csv", encoding="utf-8-sig"), valores)
        figura_local(pd.read_csv(SALIDA / "shap_local.csv", encoding="utf-8-sig", dtype={"valor": str}))
    else:
        main()
