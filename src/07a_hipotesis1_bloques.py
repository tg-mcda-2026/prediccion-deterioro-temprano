"""
07a - Primera hipótesis: bloque financiero-contractual (E1), sociodemográfico-comercial (E2) y combinado (E3), con el
mismo procedimiento de las versiones candidatas (diseño metodológico del documento, H1).

  cv <bloque>   Optuna por algoritmo (semilla 42) + validación cruzada 5x5 dentro de 2023-2024, etiqueta principal.
                Nada de 2025 se lee en este paso.
  reporte       Diferencias pareadas en la CV (Nadeau-Bengio) y ajuste con todo 2023-2024 para calificar 2025 una vez:
                AUC (DeLong y bootstrap pareado) y Brier (bootstrap pareado), Holm sobre las dos pruebas principales.
  sensibilidad  E1 solo con las condiciones del crédito (E1_condiciones, ajustado antes con cv) frente a E3 y E1, en
                la CV y en 2025. Exploratorio: fuera del contraste registrado.

Uso:     python 07a_hipotesis1_bloques.py cv E1_financiero      (y E2_sociodemografico, E3_combinado, E1_condiciones)
         python 07a_hipotesis1_bloques.py reporte
         python 07a_hipotesis1_bloques.py sensibilidad
Salidas: resultados/hipotesis/h1/
"""
import json
import sys
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score

import credito as C
import modelos as M
import protocolo as P
import utilidades_auc as U

warnings.filterwarnings("ignore")
M.NUMERICAS, M.CAT_BAJA, M.CAT_ALTA = P.NUMERICAS + C.FINANCIERAS, P.CAT_BAJA, P.CAT_ALTA
MODELOS = list(M.N_TRIALS)
SALIDA = P.RAIZ / "resultados" / "hipotesis" / "h1"
COL = P.Y
RIESGO_SEGMENTO = {"BRONCE": 3, "PLATA": 2, "ORO": 1}

E1 = ["capital", "cuota", "cuota_inicial", "cuota_inicial_pct", "tasa", "plazo", "prenda",
      "precio_moto", "propietario_moto_cliente", "familia_moto", "cilindraje"]
E2 = ["edad", "estado_civil", "estrato", "personas_a_cargo", "nivel_estudio", "independiente_informal", "ciudad",
      "zona_homologada", "departamento", "compra_fuera_depto", "log_frec_ciudad", "vol_distribuidor_rango"]
BLOQUES = {"E1_financiero": E1, "E2_sociodemografico": E2, "E3_combinado": E1 + E2}
# Sensibilidad de la revisión final (no es parte del contraste registrado): E1 solo con las condiciones del crédito,
# sin las variables del producto (precio, moto a nombre del cliente, familia de moto y cilindraje).
E1_CONDICIONES = ["capital", "cuota", "cuota_inicial", "cuota_inicial_pct", "tasa", "plazo", "prenda"]
SENSIBILIDAD = {"E1_condiciones": E1_CONDICIONES}


def con_credito(base):
    fin = C.variables_credito(base["id_obligacion"].values)
    return base.merge(fin.reset_index(), on="id_obligacion", how="left", validate="one_to_one")


def cv(bloque):
    variables = {**BLOQUES, **SENSIBILIDAD}[bloque]
    (SALIDA / "oof").mkdir(parents=True, exist_ok=True)
    archivo = SALIDA / "oof" / f"{bloque}.parquet"
    params_arch = SALIDA / f"parametros_{bloque}.json"
    dev = con_credito(P.cargar_desarrollo())  # verifica que no haya 2025
    d = dev[dev[COL].notna()].reset_index(drop=True)
    y, anio = d[COL].astype(int).values, d["anio"].values
    print(f"=== {bloque}: {len(variables)} variables, n={len(d)}, tasa={y.mean():.3f}", flush=True)
    params = json.loads(params_arch.read_text(encoding="utf-8")) if params_arch.exists() else {}
    filas = []
    for nombre in MODELOS:
        t0 = time.time()
        if nombre not in params:
            folds_opt = P.pliegues(anio, y, P.SEMILLA_OPTUNA)

            def objetivo(t):
                modelo = M.construir(nombre, M.espacio(nombre, t), variables)
                return -cross_val_score(modelo, d, y, cv=folds_opt, scoring="neg_log_loss").mean()

            estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=P.SEMILLA_OPTUNA))
            estudio.optimize(objetivo, n_trials=M.N_TRIALS[nombre])
            params[nombre] = estudio.best_params
            params_arch.write_text(json.dumps(params, indent=1), encoding="utf-8")
        for semilla in P.SEMILLAS_EVAL:
            folds = P.pliegues(anio, y, semilla)
            p = cross_val_predict(M.construir(nombre, params[nombre], variables), d, y, cv=folds,
                                  method="predict_proba")[:, 1]
            pliegue = np.empty(len(d), dtype=int)
            for k, (_, va) in enumerate(folds):
                pliegue[va] = k
            filas.append(pd.DataFrame({"id_obligacion": d["id_obligacion"].values, "anio": anio, "y": y,
                                       "semilla": semilla, "pliegue": pliegue, "modelo": nombre, "p": p}))
        print(f"  {nombre:13s} {time.time() - t0:6.0f}s", flush=True)
    largo = pd.concat(filas)
    ens = largo.groupby(["id_obligacion", "anio", "y", "semilla", "pliegue"], as_index=False)["p"].mean()
    ens["modelo"] = "ensamble"
    pd.concat([largo, ens]).to_parquet(archivo, index=False)
    r = ens.groupby("semilla").apply(lambda g: P.auc_intra(g.y, g.p, g.anio))
    print(f"  ensamble AUC dentro de cosecha {r.mean():.4f} (DE entre semillas {r.std():.4f})", flush=True)


def por_pliegue(bloque):
    o = pd.read_parquet(SALIDA / "oof" / f"{bloque}.parquet")
    o = o[o.modelo == "ensamble"]
    filas = []
    for (s, k), g in o.groupby(["semilla", "pliegue"]):
        filas.append({"semilla": s, "pliegue": k, "auc": P.auc_intra(g.y, g.p, g.anio), "brier": brier_score_loss(g.y, g.p)})
    return pd.DataFrame(filas).set_index(["semilla", "pliegue"])


def boot_brier(y, p1, p2, n=2000, semilla=0):
    """Diferencia de Brier p1 - p2 con IC95 y p bilateral por bootstrap pareado de créditos."""
    rng = np.random.default_rng(semilla)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    e = (p1 - y) ** 2 - (p2 - y) ** 2
    difs = np.array([e[rng.integers(0, len(y), len(y))].mean() for _ in range(n)])
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())
    return float(e.mean()), float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5)), float(min(1.0, p))


def reporte():
    pares = [("E3_combinado", "E1_financiero"), ("E2_sociodemografico", "E1_financiero"), ("E3_combinado", "E2_sociodemografico")]

    # ---- validación cruzada 2023-2024
    pp = {b: por_pliegue(b) for b in BLOQUES}
    res_cv = [{"bloque": b, "n_variables": len(v), "auc_intra_cosecha": pp[b].auc.mean(), "de_pliegues": pp[b].auc.std(),
               "brier": pp[b].brier.mean()} for b, v in BLOQUES.items()]
    pd.DataFrame(res_cv).to_csv(SALIDA / "cv_bloques.csv", index=False, encoding="utf-8-sig")
    dif_cv = []
    for a, b in pares:
        for m in ("auc", "brier"):
            media, ee, inf, sup, p = U.nadeau_bengio((pp[a][m] - pp[b][m]).values)
            dif_cv.append({"comparacion": f"{a} - {b}", "metrica": m, "diferencia": media, "ic_inf": inf, "ic_sup": sup,
                           "p": p, "pliegues_a_favor_de_a": int(((pp[a][m] - pp[b][m]) * (1 if m == "auc" else -1) > 0).sum())})
    pd.DataFrame(dif_cv).to_csv(SALIDA / "cv_diferencias.csv", index=False, encoding="utf-8-sig")

    # ---- ajuste con 2023-2024 y calificación única de 2025
    base = con_credito(P.cargar_reporte())  # exige decision.json
    dev, oot = base[base.periodo == "train"], base[base.periodo == "oot"].copy()
    d = dev[dev[COL].notna()]
    y_d = d[COL].astype(int).values
    for b, variables in BLOQUES.items():
        params = json.loads((SALIDA / f"parametros_{b}.json").read_text(encoding="utf-8"))
        modelos = [M.construir(n, params[n], variables).fit(d, y_d) for n in MODELOS]
        oot[f"score__{b}"] = np.mean([m.predict_proba(oot)[:, 1] for m in modelos], axis=0)
        print("calificado 2025:", b, flush=True)
    oot["score__segmento"] = oot["ref_segmento"].map(RIESGO_SEGMENTO).astype(float)
    cols = ["id_obligacion", "fecha_desembolso", COL] + [f"{o}__{P.ETIQUETA}" for o in P.OBJETIVOS_REPORTE] + \
           ["sexo", "estrato"] + \
           [c for c in oot.columns if c.startswith("score__")]
    oot[cols].to_parquet(SALIDA / "puntajes_2025_bloques.parquet", index=False)

    t = oot[oot[COL].notna()]
    y = t[COL].astype(int).values
    t1 = (t.fecha_desembolso < P.CORTE_RECALIBRACION).values
    res = []
    for b in list(BLOQUES) + ["segmento"]:
        s = t[f"score__{b}"].values
        auc, inf, sup = U.ic_auc_delong(y, s)
        fila = {"bloque": b, "n": len(y), "eventos": int(y.sum()), "auc": auc, "auc_ic_inf": inf, "auc_ic_sup": sup}
        if b != "segmento":
            a = M.recalibrar_intercepto(y[t1], s[t1])
            fila.update({"prob_media": s.mean(), "brier": brier_score_loss(y, s), "brier_constante": y.mean() * (1 - y.mean()),
                         "brier_T2_sin_recalibrar": brier_score_loss(y[~t1], s[~t1]),
                         "brier_T2_recalibrado_T1": brier_score_loss(y[~t1], M.aplicar_intercepto(s[~t1], a))})
        res.append(fila)
    pd.DataFrame(res).to_csv(SALIDA / "prueba_2025_bloques.csv", index=False, encoding="utf-8-sig")

    dif = []
    for a, b in pares:
        sa, sb = t[f"score__{a}"].values, t[f"score__{b}"].values
        _, _, d_auc, _, i1, i2, p_auc = U.delong(y, sa, sb)
        b1, b2 = U.bootstrap_pareado(y, sa, sb, n=2000)
        d_br, bi, bs, p_br = boot_brier(y, sa, sb)
        ra, rb = M.recalibrar_intercepto(y[t1], sa[t1]), M.recalibrar_intercepto(y[t1], sb[t1])
        d_br2, bi2, bs2, p_br2 = boot_brier(y[~t1], M.aplicar_intercepto(sa[~t1], ra), M.aplicar_intercepto(sb[~t1], rb))
        dif.append({"comparacion": f"{a} - {b}", "dif_auc": d_auc, "auc_ic_delong_inf": i1, "auc_ic_delong_sup": i2,
                    "auc_ic_boot_inf": b1, "auc_ic_boot_sup": b2, "p_auc": p_auc,
                    "dif_brier": d_br, "brier_ic_inf": bi, "brier_ic_sup": bs, "p_brier": p_br,
                    "dif_brier_T2_recalibrado": d_br2, "brier_T2_ic_inf": bi2, "brier_T2_ic_sup": bs2, "p_brier_T2": p_br2})
    dif = pd.DataFrame(dif)
    principal = dif.comparacion == "E3_combinado - E1_financiero"
    holm = U.holm([float(dif.loc[principal, "p_auc"].iloc[0]), float(dif.loc[principal, "p_brier"].iloc[0])])
    dif["p_holm_auc"], dif["p_holm_brier"] = np.nan, np.nan
    dif.loc[principal, "p_holm_auc"], dif.loc[principal, "p_holm_brier"] = holm[0], holm[1]
    dif.to_csv(SALIDA / "prueba_2025_diferencias.csv", index=False, encoding="utf-8-sig")

    # ---- otros objetivos con el mismo puntaje (descriptivo)
    otros = []
    for obj in P.OBJETIVOS_REPORTE:
        k = oot[f"{obj}__{P.ETIQUETA}"].notna().values
        yy = oot.loc[k, f"{obj}__{P.ETIQUETA}"].astype(int).values
        for b in list(BLOQUES) + ["segmento"]:
            auc, inf, sup = U.ic_auc_delong(yy, oot.loc[k, f"score__{b}"].values)
            otros.append({"objetivo": obj, "bloque": b, "n": int(k.sum()), "eventos": int(yy.sum()), "auc": auc,
                          "ic_inf": inf, "ic_sup": sup})
    pd.DataFrame(otros).to_csv(SALIDA / "prueba_2025_otros_objetivos.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    for f in ["cv_bloques", "cv_diferencias", "prueba_2025_bloques", "prueba_2025_diferencias", "prueba_2025_otros_objetivos"]:
        print(f"\n== {f}"); print(pd.read_csv(SALIDA / f"{f}.csv").round(4).to_string(index=False))


def sensibilidad():
    """Sensibilidad de la revisión final: E1 solo con las condiciones del crédito (E1_condiciones) frente a E3 y E1
    registrados, con los mismos pliegues de la CV 5x5 y, en 2025, con los puntajes guardados de E1 y E3 (antes comprueba
    que reajustar E1 aquí reproduce sus puntajes guardados). Exploratorio: no entra en la familia de Holm de H1."""
    b = "E1_condiciones"
    todos = {**BLOQUES, **SENSIBILIDAD}
    # ---- validación cruzada 2023-2024: mismos créditos, semillas y pliegues que los bloques registrados
    oof = {k: pd.read_parquet(SALIDA / "oof" / f"{k}.parquet") for k in [b, "E1_financiero", "E3_combinado"]}
    clave = lambda o: o[o.modelo == "ensamble"].set_index(["id_obligacion", "semilla"])["pliegue"].sort_index()
    assert clave(oof[b]).equals(clave(oof["E3_combinado"])) and clave(oof[b]).equals(clave(oof["E1_financiero"])), \
        "los pliegues no coinciden"
    pp = {k: por_pliegue(k) for k in oof}

    # ---- prueba 2025: ajuste con todo 2023-2024 y calificación única
    base = con_credito(P.cargar_reporte())
    dev, oot = base[base.periodo == "train"], base[base.periodo == "oot"].copy()
    d = dev[dev[COL].notna()]
    y_d = d[COL].astype(int).values
    for nombre_b in ["E1_financiero", b]:
        params = json.loads((SALIDA / f"parametros_{nombre_b}.json").read_text(encoding="utf-8"))
        modelos = [M.construir(n, params[n], todos[nombre_b]).fit(d, y_d) for n in MODELOS]
        oot[f"nuevo__{nombre_b}"] = np.mean([m.predict_proba(oot)[:, 1] for m in modelos], axis=0)
        print("calificado 2025:", nombre_b, flush=True)
    guardados = pd.read_parquet(SALIDA / "puntajes_2025_bloques.parquet")
    t = oot[["id_obligacion", COL, "nuevo__E1_financiero", f"nuevo__{b}"]].merge(
        guardados[["id_obligacion", "score__E1_financiero", "score__E3_combinado"]], on="id_obligacion",
        validate="one_to_one")
    control = float((t["nuevo__E1_financiero"] - t["score__E1_financiero"]).abs().max())
    print(f"control: diferencia máxima entre E1 reajustado y E1 guardado = {control:.2e}", flush=True)
    t = t.rename(columns={f"nuevo__{b}": f"score__{b}"})
    t = t[t[COL].notna()]
    y = t[COL].astype(int).values

    filas = []
    for k in [b, "E1_financiero", "E3_combinado"]:
        s = t[f"score__{k}"].values
        auc, inf, sup = U.ic_auc_delong(y, s)
        filas.append({"bloque": k, "n_variables": len(todos[k]), "auc_cv": pp[k].auc.mean(), "brier_cv": pp[k].brier.mean(),
                      "n_2025": len(y), "eventos_2025": int(y.sum()), "auc_2025": auc, "auc_ic_inf": inf,
                      "auc_ic_sup": sup, "prob_media_2025": s.mean(), "brier_2025": brier_score_loss(y, s)})
    pd.DataFrame(filas).to_csv(SALIDA / "sensibilidad_E1_condiciones_bloques.csv", index=False, encoding="utf-8-sig")

    dif = []
    for a in ["E3_combinado", "E1_financiero"]:
        m_auc, _, ia, sa_cv, p_auc_cv = U.nadeau_bengio((pp[a].auc - pp[b].auc).values)
        m_br, _, ib, sb_cv, p_br_cv = U.nadeau_bengio((pp[a].brier - pp[b].brier).values)
        sa, sb = t[f"score__{a}"].values, t[f"score__{b}"].values
        _, _, d_auc, _, i1, i2, p_auc = U.delong(y, sa, sb)
        b1, b2 = U.bootstrap_pareado(y, sa, sb, n=2000)
        d_br, bi, bs, p_br = boot_brier(y, sa, sb)
        dif.append({"comparacion": f"{a} - {b}",
                    "dif_auc_cv": m_auc, "auc_cv_ic_inf": ia, "auc_cv_ic_sup": sa_cv, "p_auc_cv": p_auc_cv,
                    "pliegues_a_favor_de_a": int(((pp[a].auc - pp[b].auc) > 0).sum()),
                    "dif_brier_cv": m_br, "brier_cv_ic_inf": ib, "brier_cv_ic_sup": sb_cv, "p_brier_cv": p_br_cv,
                    "dif_auc_2025": d_auc, "auc_ic_delong_inf": i1, "auc_ic_delong_sup": i2,
                    "auc_ic_boot_inf": b1, "auc_ic_boot_sup": b2, "p_auc_2025": p_auc,
                    "dif_brier_2025": d_br, "brier_ic_inf": bi, "brier_ic_sup": bs, "p_brier_2025": p_br})
    pd.DataFrame(dif).to_csv(SALIDA / "sensibilidad_E1_condiciones_diferencias.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for f in ["sensibilidad_E1_condiciones_bloques", "sensibilidad_E1_condiciones_diferencias"]:
        print(f"\n== {f}"); print(pd.read_csv(SALIDA / f"{f}.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "cv":
        cv(sys.argv[2])
    elif sys.argv[1] == "sensibilidad":
        sensibilidad()
    else:
        reporte()
