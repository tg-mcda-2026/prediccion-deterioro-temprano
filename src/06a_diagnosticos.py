"""
06a - Diagnósticos que NO cambian la decisión (protocolo, sección 10). Todo con 2023-2024, salvo el PSI frente a 2025,
que solo compara covariables y puntajes (sin etiquetas).

  A. Sobreajuste: AUC aparente (ajuste y evaluación con los mismos créditos) frente al AUC de CV, por algoritmo; el
     AUC de CV es el de la selección (promedio de los 25 pliegues). Solo esta parte: --solo-sobreajuste.
  B. Deriva frente a azar: entrenar con 2023 y probar con 2024, frente a un holdout aleatorio del mismo tamaño
     dentro de 2023-2024 (20 repeticiones).
  C. Efecto de cosecha en el ajuste (solo 2023-2024): logit(y) = logit(puntaje OOF) + año + puntaje x año.
  D. Curva de aprendizaje con CV dentro de 2023-2024: en cada pliegue se entrena con el 25/50/75/100 % del
     entrenamiento y se evalúa en el mismo pliegue de validación; media y DE en los 25 pliegues.
     Algoritmos: logística, tarjeta WoE, random forest, LightGBM y XGBoost (sin CatBoost ni EBM, por tiempo).
  E. PSI corregido (binarias como categóricas, categorías nuevas incluidas, raras agrupadas) con línea base por azar.
  F. Prueba de fuga: la etiqueta permutada debe dar AUC cercano a 0,5.
Salidas: resultados/protocolo/diagnosticos/
"""
from pathlib import Path
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict

import protocolo as P
import utilidades_auc as U

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parents[1]
import modelos as M
M.NUMERICAS, M.CAT_BAJA, M.CAT_ALTA = P.NUMERICAS, P.CAT_BAJA, P.CAT_ALTA
MODELOS = list(M.N_TRIALS)
RAPIDOS = ["logistica", "scorecard", "random_forest", "lightgbm", "xgboost"]
SALIDA = P.SALIDA / "diagnosticos"
COL = P.Y
lp = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def ensamble(version, variables, params, Xtr, ytr, modelos=None):
    ms = [M.construir(n, params[f"{version}|{n}"], variables).fit(Xtr, ytr) for n in (modelos or MODELOS)]
    return lambda X: np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0), ms


def psi(a, b, base_cats=None):
    """PSI con unión de categorías; numéricas en deciles de `a`."""
    if a.dtype == object or a.nunique() <= 12:
        a, b = a.astype(str), b.astype(str)
        top = a.value_counts().index[:20]
        a, b = a.where(a.isin(top), "OTRAS"), b.where(b.isin(top), "OTRAS")
        cats = sorted(set(a) | set(b))
        e = a.value_counts(normalize=True).reindex(cats).fillna(0).values
        o = b.value_counts(normalize=True).reindex(cats).fillna(0).values
    else:
        cortes = np.unique(np.nanquantile(a, np.linspace(0, 1, 11)))
        cortes[0], cortes[-1] = -np.inf, np.inf
        e = pd.cut(a, cortes).value_counts(normalize=True, sort=False).values
        o = pd.cut(b, cortes).value_counts(normalize=True, sort=False).values
    e, o = np.clip(e, 1e-4, None), np.clip(o, 1e-4, None)
    return float(np.sum((o - e) * np.log(o / e)))


def sobreajuste(versiones, cand, params, d, y, anio):
    """A. AUC aparente (ajuste y evaluación con los mismos créditos) frente al AUC de validación cruzada, calculado como
    en la selección (04b_decision.py): AUC dentro de cosecha de cada uno de los 25 pliegues, promediado."""
    filas = []
    for v in versiones:
        oof = pd.read_parquet(P.SALIDA / "oof" / f"{v}__{P.ETIQUETA}.parquet")
        pred, ms = ensamble(v, cand[v], params, d, y)
        for nombre, m in list(zip(MODELOS, ms)) + [("ensamble", None)]:
            p_ap = pred(d) if m is None else m.predict_proba(d)[:, 1]
            o = oof[oof.modelo == nombre]
            cv = o.groupby(["semilla", "pliegue"]).apply(lambda g: P.auc_intra(g.y, g.p, g.anio)).mean()
            filas.append({"version": v, "modelo": nombre, "auc_aparente": P.auc_intra(y, p_ap, anio),
                          "auc_cv": cv, "optimismo": P.auc_intra(y, p_ap, anio) - cv})
        print("A", v, flush=True)
    pd.DataFrame(filas).to_csv(SALIDA / "A_sobreajuste.csv", index=False, encoding="utf-8-sig")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    decision = json.loads(P.DECISION.read_text(encoding="utf-8"))
    params = json.loads((P.SALIDA / "parametros.json").read_text(encoding="utf-8"))
    tabla = pd.read_csv(P.SALIDA / "decision_tabla.csv")
    tabla = tabla[tabla.etiqueta == P.ETIQUETA]
    elegida, mejor = decision["version"], tabla.loc[tabla.es_mejor, "version"].iloc[0]
    cand = P.candidatas()
    versiones = list(dict.fromkeys([elegida, mejor, "C3_V2p"]))
    dev = P.cargar_desarrollo()
    d = dev[dev[COL].notna()].reset_index(drop=True)
    y, anio = d[COL].astype(int).values, d["anio"].values

    # A. Sobreajuste: aparente frente a CV
    sobreajuste(versiones, cand, params, d, y, anio)
    if "--solo-sobreajuste" in sys.argv:
        return

    # B. Deriva: 2023 -> 2024 frente a holdout aleatorio del mismo tamaño
    filas = []
    for v in versiones:
        tr, te = anio == 2023, anio == 2024
        pred, _ = ensamble(v, cand[v], params, d[tr], y[tr])
        filas.append({"version": v, "esquema": "temporal 2023 -> 2024", "repeticion": 0,
                      "auc": U.ic_auc_delong(y[te], pred(d[te]))[0]})
        sss = StratifiedShuffleSplit(20, train_size=tr.sum(), test_size=te.sum(), random_state=0)
        for r, (i_tr, i_te) in enumerate(sss.split(np.zeros(len(y)), anio * 10 + y)):
            pred, _ = ensamble(v, cand[v], params, d.iloc[i_tr], y[i_tr], RAPIDOS)
            filas.append({"version": v, "esquema": "aleatorio (mismo tamaño)", "repeticion": r,
                          "auc": P.auc_intra(y[i_te], pred(d.iloc[i_te]), anio[i_te])})
        print("B", v, flush=True)
    pd.DataFrame(filas).to_csv(SALIDA / "B_deriva_vs_azar.csv", index=False, encoding="utf-8-sig")

    # C. Efecto de cosecha (solo 2023-2024)
    oof = pd.read_parquet(P.SALIDA / "oof" / f"{elegida}__{P.ETIQUETA}.parquet")
    oof = oof[oof.modelo == "ensamble"].groupby(["id_obligacion", "anio", "y"], as_index=False)["p"].mean()
    dd = pd.DataFrame({"y": oof.y, "lp": lp(oof.p), "anio": oof.anio.astype(str)})
    m0, m1, m2 = (smf.logit(f, dd).fit(disp=0) for f in ["y ~ lp", "y ~ lp + C(anio)", "y ~ lp * C(anio)"])
    pd.DataFrame([{"prueba": "nivel (año)", "lr": 2 * (m1.llf - m0.llf), "p": chi2.sf(2 * (m1.llf - m0.llf), 1)},
                  {"prueba": "orden (puntaje x año)", "lr": 2 * (m2.llf - m1.llf), "p": chi2.sf(2 * (m2.llf - m1.llf), 1)},
                  *[{"prueba": f"coef {k}", "coef": m2.params[k], "ic_inf": m2.conf_int().loc[k, 0],
                     "ic_sup": m2.conf_int().loc[k, 1]} for k in m2.params.index]]
                 ).to_csv(SALIDA / "C_efecto_cosecha_dev.csv", index=False, encoding="utf-8-sig")
    print("C listo", flush=True)

    # D. Curva de aprendizaje con CV dentro de 2023-2024
    filas = []
    for semilla in P.SEMILLAS_EVAL:
        for k, (tr, va) in enumerate(P.pliegues(anio, y, semilla)):
            rng = np.random.default_rng(semilla * 10 + k)
            for frac in (0.25, 0.5, 0.75, 1.0):
                if frac < 1:
                    sub = StratifiedShuffleSplit(1, train_size=frac, random_state=int(rng.integers(1e6)))
                    idx = tr[next(sub.split(np.zeros(len(tr)), anio[tr] * 10 + y[tr]))[0]]
                else:
                    idx = tr
                for v in versiones:
                    pred, _ = ensamble(v, cand[v], params, d.iloc[idx], y[idx], RAPIDOS)
                    filas.append({"semilla": semilla, "pliegue": k, "fraccion": frac, "n_entrenamiento": len(idx),
                                  "version": v, "auc": P.auc_intra(y[va], pred(d.iloc[va]), anio[va])})
        print("D semilla", semilla, flush=True)
        pd.DataFrame(filas).to_csv(SALIDA / "D_curva_aprendizaje_cv.csv", index=False, encoding="utf-8-sig")
    C = pd.DataFrame(filas)
    res = C.groupby(["version", "fraccion"]).agg(n=("n_entrenamiento", "mean"), auc=("auc", "mean"),
                                                  de=("auc", "std")).reset_index()
    piv = C.pivot_table(index=["semilla", "pliegue", "fraccion"], columns="version", values="auc")
    difs = []
    for v in versiones[1:]:
        dif = (piv[v] - piv[versiones[0]]).groupby(level="fraccion").agg(["mean", "std"]).reset_index()
        difs.append(dif.rename(columns={"mean": f"dif_vs_{versiones[0]}", "std": "dif_de"}).assign(version=v))
    if difs:
        res = res.merge(pd.concat(difs), on=["version", "fraccion"], how="left")
    res.to_csv(SALIDA / "D_curva_resumen.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for v in versiones:
        r = res[res.version == v]
        ax.errorbar(r["n"], r["auc"], yerr=r["de"], marker="o", capsize=3, label=v)
    ax.set(xlabel="Créditos de entrenamiento", ylabel="AUC dentro de cosecha (CV)",
           title="Curva de aprendizaje (CV 2023-2024, media ± DE)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(SALIDA / "D_curva_aprendizaje_cv.png", dpi=130); plt.close(fig)

    # E. PSI corregido con línea base por azar (covariables; el puntaje de 2025 si el reporte ya se corrió)
    base = P.cargar_reporte()
    g23, g24, g25 = (base[base.anio == a] for a in (2023, 2024, 2025))
    ref = base[base.anio <= 2024]
    rng = np.random.default_rng(0)
    filas = []
    for v in cand[elegida]:
        azar = []
        for _ in range(50):
            i = rng.permutation(len(ref))
            azar.append(psi(ref[v].iloc[i[:len(g23)]], ref[v].iloc[i[len(g23):]]))
        filas.append({"variable": v, "psi_2023_2024": psi(g23[v], g24[v]), "psi_2023_24_a_2025": psi(ref[v], g25[v]),
                      "azar_media": np.mean(azar), "azar_p95": np.percentile(azar, 95)})
    archivo = P.SALIDA / "reporte_2025" / "puntajes_2025.parquet"
    if archivo.exists():
        s25 = pd.read_parquet(archivo)[f"score__{elegida}"]
        filas.append({"variable": "PUNTAJE (OOF 2023-24 frente a 2025)", "psi_2023_24_a_2025": psi(oof.p, s25),
                      "psi_2023_2024": psi(oof.p[oof.anio == 2023], oof.p[oof.anio == 2024])})
    pd.DataFrame(filas).to_csv(SALIDA / "E_psi.csv", index=False, encoding="utf-8-sig")
    print("E listo", flush=True)

    # F. Prueba de fuga con la etiqueta permutada
    filas = []
    folds = P.pliegues(anio, y, 1)
    for r in range(5):
        yp = np.random.default_rng(100 + r).permutation(y)
        preds = []
        for n in MODELOS:
            preds.append(cross_val_predict(M.construir(n, params[f"{elegida}|{n}"], cand[elegida]), d, yp, cv=folds,
                                           method="predict_proba")[:, 1])
        for n, p in zip(MODELOS + ["ensamble"], preds + [np.mean(preds, axis=0)]):
            filas.append({"repeticion": r, "modelo": n, "auc_etiqueta_permutada": P.auc_intra(yp, p, anio)})
    pd.DataFrame(filas).to_csv(SALIDA / "F_prueba_fuga.csv", index=False, encoding="utf-8-sig")
    print("F listo", flush=True)

    pd.set_option("display.width", 220)
    for f in ["A_sobreajuste", "C_efecto_cosecha_dev", "D_curva_resumen", "E_psi"]:
        print(f"\n== {f}"); print(pd.read_csv(SALIDA / f"{f}.csv").round(4).to_string(index=False))
    b = pd.read_csv(SALIDA / "B_deriva_vs_azar.csv")
    print(b.groupby(["version", "esquema"])["auc"].agg(["mean", "std"]).round(4))
    print(pd.read_csv(SALIDA / "F_prueba_fuga.csv").groupby("modelo")["auc_etiqueta_permutada"].agg(["mean", "min", "max"]).round(3))


if __name__ == "__main__":
    main()
