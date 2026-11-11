"""
Protocolo de selección (descrito en el diseño metodológico del documento).

- 2025 queda sellado: cargar_desarrollo() solo devuelve desembolsos de 2023-2024 y lo verifica.
- 2025 solo se carga con cargar_reporte(), que exige que exista resultados/protocolo/decision.json.
- Variables del lugar calculadas con la cartera de motos anterior a 2025, sin el propio crédito y sin usar el
  distribuidor como respaldo (tienda sin historia -> compra_fuera_depto desconocida).
- Zona homologada: la de 2025 de la ciudad; si el nombre existe en varios departamentos, la del municipio (nombre y
  departamento), para no darle la zona del homónimo.
- Etiqueta del evento (columnas __B_todos): los créditos que salen del panel sin deterioro son pagos anticipados
  (terminaron de pagar antes de su fecha) y cuentan como buenos pagadores; los que salen con el deterioro ya registrado son eventos.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import divipola as D
import limpieza as L

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "resultados" / "protocolo"
DECISION = SALIDA / "decision.json"
CORTE_OOT = pd.Timestamp("2025-01-01")
CORTE_RECALIBRACION = pd.Timestamp("2025-04-01")

OBJETIVO = "mora30_6m"
OBJETIVOS_REPORTE = ["mora30_3m", "mora60_6m", "mora90_6m"]  # monitoreo y confirmación, con el mismo puntaje
ETIQUETA = "B_todos"                    # pagos anticipados (salidas del panel sin deterioro) como buenos pagadores
ETIQUETAS = [ETIQUETA]
Y = f"{OBJETIVO}__{ETIQUETA}"           # columna del evento principal
SEMILLA_OPTUNA = 42
SEMILLAS_EVAL = [1, 2, 3, 4, 5]
N_PLIEGUES = 5

# --------------------------------------------------------------------------- variables
PERSONA = ["edad", "estado_civil", "estrato", "personas_a_cargo", "nivel_estudio", "independiente_informal", "ciudad"]
LUGAR = ["zona_homologada", "departamento", "compra_fuera_depto", "log_frec_ciudad", "vol_distribuidor_rango"]
PRODUCTO = ["familia_moto", "tipo_uso_moto", "cilindraje", "precio_moto", "propietario_moto_cliente"]
PERMITIDAS = PERSONA + LUGAR + PRODUCTO
EXCLUIDAS = ["sexo", "codigo_distribuidor", "codigo_tienda", "canal_pct_moto_otro", "mujer_scooter",
             "indice_mora_ciudad", "indice_mora_tienda"]  # más todo lo financiero, que no está en la base
SOLO_AUDITORIA = ["sexo"]

NUMERICAS = ["edad", "estrato", "personas_a_cargo", "independiente_informal", "cilindraje", "precio_moto",
             "propietario_moto_cliente", "compra_fuera_depto", "log_frec_ciudad"]
CAT_BAJA = ["sexo", "estado_civil", "nivel_estudio", "familia_moto", "tipo_uso_moto", "zona_homologada",
            "vol_distribuidor_rango"]
CAT_ALTA = ["ciudad", "departamento", "codigo_distribuidor"]  # codificación por la tasa del evento, dentro del pliegue

V2P = ["edad", "estado_civil", "ciudad", "independiente_informal", "nivel_estudio", "familia_moto"]
CANDIDATAS_FIJAS = {
    "C1_completo": PERMITIDAS,
    "C3_V2p": V2P,
    "C4_V2p_compra": V2P + ["compra_fuera_depto"],
    "C5_V3p": V2P + ["compra_fuera_depto", "log_frec_ciudad", "vol_distribuidor_rango"],
    "C6_V5p": V2P + ["compra_fuera_depto", "log_frec_ciudad", "vol_distribuidor_rango", "departamento"],
}
# Ablación: costo de quitar el sexo de C3 (informativa, no es candidata)
ABLACION = {"A_C3_con_sexo": ["sexo"] + V2P}
# Sensibilidad de la revisión final (exploratoria; no es candidata ni entra en la decisión): C3 sin el estado civil
SENSIBILIDAD = {"S_C3_sin_estado_civil": [v for v in V2P if v != "estado_civil"]}
TABLAS_EXTERNAS = {"departamento"}  # DIVIPOLA
REFERENCIA = {"compra_fuera_depto", "log_frec_ciudad", "vol_distribuidor_rango", "departamento"}

RANGOS_VOLUMEN = [(-1, 0, "NUEVO"), (0, 99, "1-99"), (99, 499, "100-499"), (499, 1499, "500-1499"),
                  (1499, np.inf, "1500+")]


def candidatas():
    """Lista cerrada: las fijas más C2 (resultado de la selección hacia atrás)."""
    c = dict(CANDIDATAS_FIJAS)
    archivo = SALIDA / "seleccion_variables.json"
    if archivo.exists():
        c["C2_seleccionado"] = json.loads(archivo.read_text(encoding="utf-8"))["variables"]
    return c


def simplicidad(variables):
    """Orden de simplicidad: número de variables; a igual número, sin tablas externas ni periodo de referencia."""
    return (len(variables), len(set(variables) & TABLAS_EXTERNAS), len(set(variables) & REFERENCIA))


# --------------------------------------------------------------------------- variables del lugar
def _cartera_c0():
    df = pd.read_parquet(RAIZ / "datos" / "cartera.parquet")
    s = df[(df["Corte"] < df["Corte"].max()) & (df["Tipo"] == "01CR") & (df["Productos"] == "Motocicletas")
           & (df["Mes_Cosecha"] == 0)].drop_duplicates("Numero_Encriptado").set_index("Numero_Encriptado")
    p = pd.DataFrame(index=s.index)
    p["desembolso"] = L.fecha_excel(s["Fecha desembolso"])
    p["ciudad"] = L.normalizar_texto(s["Ciudad"])
    p["tienda"] = pd.to_numeric(s["Codigo de la tienda"], errors="coerce")
    p["distribuidor"] = L.codigo_entero(s["Codigo del distribuidor"]).astype(float)
    p["departamento"] = D.asignar_departamento(s["Ciudad"], s["Nombre Zona"]).values
    return p


def _zona_municipio_2025(p):
    """Zona de la estructura comercial de 2025 por municipio (nombre y departamento), con las mismas etiquetas de
    limpieza.zona_homologada, que agrupa solo por nombre y a los nombres repetidos les da la zona del homónimo."""
    df = pd.read_parquet(RAIZ / "datos" / "cartera.parquet")
    c0 = df[(df["Tipo"] == "01CR") & (df["Productos"] == "Motocicletas") & (df["Mes_Cosecha"] == 0)]
    d = c0[L.fecha_excel(c0["Corte"]).dt.year == 2025]
    zona = L.codigo_entero(d["Zona"])
    nombres = L.normalizar_texto(d.groupby(zona)["Nombre Zona"].agg(lambda s: s.mode().iloc[0]))
    validas = nombres[~nombres.str.startswith("SIN ASIGNAR")]
    etiquetas = {z: f"Z{z:02d} {n.split('-')[0].strip()}" for z, n in validas.items()}
    r = pd.DataFrame({"zona": zona.values, "ciudad": p["ciudad"].reindex(d["Numero_Encriptado"]).values,
                      "departamento": p["departamento"].reindex(d["Numero_Encriptado"]).values})
    r = r[r["zona"].isin(validas.index)].dropna(subset=["ciudad", "departamento"])
    return r.groupby(["ciudad", "departamento"])["zona"].agg(lambda s: s.mode().iloc[0]).map(etiquetas)


def _moda_sin_propio(ref, grupo, valor, filas):
    """Moda de `valor` por `grupo` en la referencia, quitando una vez el propio registro de cada fila de `filas`."""
    cuenta = ref.dropna(subset=[grupo, valor]).groupby([grupo, valor]).size()
    salida = pd.Series(np.nan, index=filas.index, dtype=object)
    propios = filas.index.isin(ref.index)
    for idx, (g, v) in filas[[grupo, valor]].iterrows():
        if pd.isna(g) or g not in cuenta.index.get_level_values(0):
            continue
        c = cuenta.loc[g].copy()
        if propios[filas.index.get_loc(idx)] and pd.notna(v) and v in c.index:
            c.loc[v] -= 1
        c = c[c > 0]
        if len(c):
            salida.loc[idx] = c.sort_index().idxmax()  # igualdad de conteos: orden alfabético, reproducible
    return salida


def variables_lugar(ids):
    p = _cartera_c0()
    ref = p[p["desembolso"] < CORTE_OOT]
    # Departamento: DIVIPOLA + zona; si sigue ambiguo, el departamento de la tienda (el más frecuente de sus clientes en
    # la referencia) cuando es uno de los candidatos del nombre; si aún no se resuelve, el más frecuente de la ciudad
    dep_tienda_ref = ref.dropna(subset=["departamento", "tienda"]).groupby("tienda")["departamento"] \
        .agg(lambda x: x.mode().sort_values().iloc[0])
    falta = p["departamento"].isna() & p["ciudad"].notna()
    cands = pd.Series(D.candidatos_departamento(p.loc[falta, "ciudad"]), index=p.index[falta])
    dt = p.loc[falta, "tienda"].map(dep_tienda_ref)
    p.loc[falta, "departamento"] = [d if isinstance(d, str) and d in c else np.nan for c, d in zip(cands, dt)]
    moda_dep = ref.dropna(subset=["departamento"]).groupby("ciudad")["departamento"].agg(lambda x: x.mode().iloc[0])
    p["departamento"] = p["departamento"].fillna(p["ciudad"].map(moda_dep))
    b = p.loc[ids].copy()
    propio = b.index.isin(ref.index).astype(int)

    ciudad_tienda = _moda_sin_propio(ref, "tienda", "ciudad", b)
    dep_ciudad = p.loc[p.index.isin(ref.index)].dropna(subset=["departamento"]).groupby("ciudad")["departamento"] \
        .agg(lambda x: x.mode().iloc[0])
    dep_tienda = ciudad_tienda.map(dep_ciudad)
    out = pd.DataFrame(index=b.index)
    out["departamento"] = b["departamento"]
    out["compra_fuera_depto"] = np.where(b["departamento"].isna() | dep_tienda.isna(), np.nan,
                                         (b["departamento"] != dep_tienda).astype(float))
    frec = b["ciudad"].map(ref.groupby("ciudad").size()).fillna(0) - propio
    out["log_frec_ciudad"] = np.log1p(frec.clip(lower=0))
    vol = b["distribuidor"].map(ref.groupby("distribuidor").size()).fillna(0) - propio
    out["vol_distribuidor"] = vol.clip(lower=0)
    out["vol_distribuidor_rango"] = pd.cut(out["vol_distribuidor"], [r[0] for r in RANGOS_VOLUMEN] + [np.inf],
                                           labels=[r[2] for r in RANGOS_VOLUMEN]).astype(str)
    # Zona homologada de los nombres que existen en varios departamentos: la de 2025 del municipio (nombre y
    # departamento); desconocida si ese municipio no tiene créditos en 2025, como cualquier otro (_base_completa)
    zonas = _zona_municipio_2025(p)
    repetido = pd.Series(D.candidatos_departamento(b["ciudad"]), index=b.index).map(len) > 1
    out["_corrige_zona"] = repetido & b["departamento"].notna()
    out["_zona_municipio"] = [zonas.get((c, d), np.nan) for c, d in zip(b["ciudad"], b["departamento"])]
    return out


# --------------------------------------------------------------------------- etiquetas
def agregar_etiquetas(base):
    """Evento con los pagos anticipados como buenos: los que salen del panel sin deterioro (al día, con mora menor de
    30 días o en el mismo mes del desembolso) valen 0; los que salen con el deterioro ya registrado conservan el 1."""
    for obj in ["mora30_6m", "mora60_6m", "mora90_6m", "mora30_3m"]:
        h = 3 if obj.endswith("3m") else 6
        estado = base[f"estado_{h}m"]
        sale_sin_deterioro = base[obj].isna() & estado.isin(["sale_al_dia", "sale_mora_menor_30", "sale_sin_exposicion"])
        base[f"{obj}__B_todos"] = base[obj].where(~sale_sin_deterioro, 0.0)
    return base


# --------------------------------------------------------------------------- carga
_CACHE = RAIZ / "datos" / "base_protocolo.parquet"


def _base_completa():
    """Base con variables del lugar y etiquetas. Se guarda en datos/base_protocolo.parquet (no toca base_modelo)."""
    if _CACHE.exists():
        return pd.read_parquet(_CACHE)
    base = pd.read_parquet(RAIZ / "datos" / "base_modelo.parquet")
    lugar = variables_lugar(base["id_obligacion"].values)
    lugar.index = base.index
    corrige, zona_municipio = lugar.pop("_corrige_zona").astype(bool), lugar.pop("_zona_municipio")
    base.loc[corrige, "zona_homologada"] = zona_municipio[corrige]
    base = agregar_etiquetas(base.join(lugar))
    base["anio"] = base["fecha_desembolso"].dt.year
    base.to_parquet(_CACHE, index=False)
    return base


def _preparar(base):
    for c in CAT_BAJA + CAT_ALTA:
        base[c] = base[c].astype("string").fillna("FALTANTE").astype(str)
    return base


def cargar_desarrollo():
    """Solo desembolsos 2023-2024. 2025 nunca sale de aquí."""
    base = _base_completa()
    dev = base[base["fecha_desembolso"] < CORTE_OOT].copy()
    assert dev["fecha_desembolso"].max() < CORTE_OOT, "2025 está sellado"
    assert set(dev["anio"].unique()) <= {2023, 2024}
    return _preparar(dev)


def cargar_reporte():
    """Toda la base, incluido 2025. Solo se permite después de registrar la decisión."""
    if not DECISION.exists():
        raise RuntimeError("2025 está sellado: primero hay que registrar resultados/protocolo/decision.json")
    base = _base_completa().copy()
    base["periodo"] = np.where(base["fecha_desembolso"] < CORTE_OOT, "train", "oot")
    return _preparar(base)


# --------------------------------------------------------------------------- validación cruzada y métrica
def pliegues(anio, y, semilla):
    """5 pliegues estratificados por año x evento."""
    estrato = np.asarray(anio) * 10 + np.asarray(y).astype(int)
    return list(StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=semilla).split(np.zeros(len(y)), estrato))


def auc_intra(y, p, anio):
    """AUC dentro de cada cosecha, promedio ponderado por número de créditos."""
    y, p, anio = np.asarray(y), np.asarray(p), np.asarray(anio)
    aucs, pesos = [], []
    for a in np.unique(anio):
        m = anio == a
        if y[m].min() != y[m].max():
            aucs.append(roc_auc_score(y[m], p[m]))
            pesos.append(m.sum())
    return float(np.average(aucs, weights=pesos))
