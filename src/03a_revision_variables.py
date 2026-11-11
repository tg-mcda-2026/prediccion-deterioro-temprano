"""
03a - Revisión univariada de las variables, SOLO con 2023-2024 (reemplaza a la revisión anterior al protocolo, que usó 2025).

Para cada variable permitida (y las excluidas, como referencia): nulos, valor dominante, IV y AUC univariado con
codificación por tasa del evento dentro del pliegue (CV 5x5 por año x evento), AUC dentro de 2023 y de 2024 por
separado, y V de Cramér con el año (cuánto distingue la cosecha). También la correlación entre variables.
Es descriptiva: la selección se hace en 03b_seleccion_variables.py (eliminación hacia atrás).

Salidas: resultados/protocolo/revision_univariada_dev.csv, correlacion_dev.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

import protocolo as P

Y = P.Y


def discretizar(s, q=10):
    if s.dtype == object or s.nunique() <= 12:
        return s.astype(str)
    cortes = np.unique(np.nanquantile(s, np.linspace(0, 1, q + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    return pd.cut(s, cortes, duplicates="drop").astype(str).where(s.notna(), "NA")


def iv(b, y):
    t = pd.crosstab(b, y) + 0.5
    pb, pg = t[1] / t[1].sum(), t[0] / t[0].sum()
    return float(((pb - pg) * np.log(pb / pg)).sum()), float(chi2_contingency(t - 0.5 + 1e-9)[1])


def cramer(a, c):
    t = pd.crosstab(a, c)
    if min(t.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.values.sum(); r, k = t.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-9, min(kc - 1, rc - 1))))


def codificar(btr, ytr, bte, m=20):
    prior = ytr.mean()
    g = pd.DataFrame({"b": btr.values, "y": ytr.values}).groupby("b")["y"].agg(["sum", "count"])
    return bte.map((g["sum"] + m * prior) / (g["count"] + m)).fillna(prior).values


def main():
    P.SALIDA.mkdir(parents=True, exist_ok=True)
    dev = P.cargar_desarrollo()
    d = dev[dev[Y].notna()].reset_index(drop=True)
    y, anio = d[Y].astype(int), d["anio"].values
    variables = P.PERMITIDAS + ["sexo", "codigo_distribuidor"]
    filas = []
    for v in variables:
        s = d[v].replace("FALTANTE", np.nan) if d[v].dtype == object else d[v]
        b = discretizar(s)
        valor_iv, p = iv(b, y)
        aucs_cv = []
        for semilla in P.SEMILLAS_EVAL:
            oof = np.zeros(len(d))
            for tr, te in P.pliegues(anio, y, semilla):
                oof[te] = codificar(b.iloc[tr], y.iloc[tr], b.iloc[te])
            aucs_cv.append(P.auc_intra(y, oof, anio))
        fila = {"variable": v, "grupo": ("excluida (solo auditoría)" if v in ("sexo", "codigo_distribuidor") else
                                        "persona" if v in P.PERSONA else "lugar" if v in P.LUGAR else "producto"),
                "pct_nulos": float(s.isna().mean()), "n_categorias": int(s.nunique()),
                "pct_valor_dominante": float(s.value_counts(normalize=True, dropna=False).iloc[0]),
                "iv": valor_iv, "p_chi2": p, "auc_cv_intra_cosecha": float(np.mean(aucs_cv)),
                "auc_cv_de": float(np.std(aucs_cv)), "v_cramer_anio": cramer(b, anio)}
        for a in (2023, 2024):
            m = anio == a
            oof = np.zeros(m.sum())
            dd, yy = b[m].reset_index(drop=True), y[m].reset_index(drop=True)
            for tr, te in P.pliegues(np.full(m.sum(), a), yy, 1):
                oof[te] = codificar(dd.iloc[tr], yy.iloc[tr], dd.iloc[te])
            fila[f"auc_cv_{a}"] = roc_auc_score(yy, oof)
        filas.append(fila)
    R = pd.DataFrame(filas).sort_values(["grupo", "auc_cv_intra_cosecha"], ascending=[True, False])
    R.to_csv(P.SALIDA / "revision_univariada_dev.csv", index=False, encoding="utf-8-sig")

    B = {v: discretizar(d[v].replace("FALTANTE", np.nan) if d[v].dtype == object else d[v]) for v in variables}
    V = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
    for i, a in enumerate(variables):
        for c in variables[i + 1:]:
            V.loc[a, c] = V.loc[c, a] = cramer(B[a], B[c])
    V.to_csv(P.SALIDA / "correlacion_dev.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False))
    print("\nPares con V de Cramér >= 0,7:")
    for i, a in enumerate(variables):
        for c in variables[i + 1:]:
            if V.loc[a, c] >= 0.7:
                print(f"  {a:24s} {c:24s} {V.loc[a, c]:.2f}")


if __name__ == "__main__":
    main()
