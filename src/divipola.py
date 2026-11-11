"""
Asignación del departamento (código DIVIPOLA del DANE) a partir del nombre de la ciudad.

Fuente: datos/divipola.csv, descargado de datos.gov.co ("DIVIPOLA - Códigos municipios", 1.122 municipios).
- Los nombres se comparan normalizados (mayúsculas, sin tildes ni puntuación).
- ALIAS cubre las formas de la base que no coinciden con el nombre oficial (Cali, Cúcuta, Cartagena...).
- Cuando un nombre existe en varios departamentos (Rionegro, Mosquera, Caldas...), se resuelve con la zona
  comercial registrada en el mismo crédito: solo cuentan las partes de la zona que son listas de regiones, porque la
  zona también trae el nombre del gerente ("CUNDINAMARCA/ BOYACA/ META- CESAR": Cesar es una persona).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import limpieza as L

RAIZ = Path(__file__).resolve().parents[1]

ALIAS = {
    "BOGOTA": "BOGOTA D C", "CALI": "SANTIAGO DE CALI", "CUCUTA": "SAN JOSE DE CUCUTA",
    "CARTAGENA": "CARTAGENA DE INDIAS", "SANTAFE DE ANTIOQUIA": "SANTA FE DE ANTIOQUIA", "DON MATIAS": "DONMATIAS",
    "TUMACO": "SAN ANDRES DE TUMACO", "MARIQUITA": "SAN SEBASTIAN DE MARIQUITA",
    "PURISIMA": "PURISIMA DE LA CONCEPCION", "BELEN DE BAJIRA": "NUEVO BELEN DE BAJIRA",
    "SAN JUAN DE RIO SECO": "SAN JUAN DE RIOSECO", "PIENDAMO": "PIENDAMO TUNIA", "SOTARA": "SOTARA PAISPAMBA",
    "LEGUIZAMO": "PUERTO LEGUIZAMO", "SAN JOSE DE LA FRAGUA": "SAN JOSE DEL FRAGUA",
    "TOLUVIEJO": "SAN JOSE DE TOLUVIEJO",
}
# Nombres que en la base corresponden a varios municipios oficiales (se resuelven con la zona)
VARIOS = {"SAN VICENTE": ["SAN VICENTE FERRER", "SAN VICENTE DEL CAGUAN", "SAN VICENTE DE CHUCURI"],
          # en la base, «San Pedro» también es San Pedro de los Milagros (Antioquia) y San Pedro de Urabá
          "SAN PEDRO": ["SAN PEDRO", "SAN PEDRO DE LOS MILAGROS", "SAN PEDRO DE URABA"]}

# Región de la zona comercial -> departamentos (nombres normalizados)
REGIONES = {
    "ANTIOQUIA": ["ANTIOQUIA"], "URABA": ["ANTIOQUIA", "CHOCO"], "BAJO CAUCA": ["ANTIOQUIA"],
    "CAQUETA": ["CAQUETA"], "PUTUMAYO": ["PUTUMAYO"], "CUNDINAMARCA": ["CUNDINAMARCA", "BOGOTA D C"],
    "BOYACA": ["BOYACA"], "META": ["META"], "CASANARE": ["CASANARE"], "VALLE": ["VALLE DEL CAUCA"],
    "VALLE DEL CAUCA": ["VALLE DEL CAUCA"], "CAUCA": ["CAUCA"], "HUILA": ["HUILA"], "NARINO": ["NARINO"],
    "SANTANDERES": ["SANTANDER", "NORTE DE SANTANDER"], "CESAR": ["CESAR"], "BOLIVAR": ["BOLIVAR"],
    "ATLANTICO": ["ATLANTICO"], "EJE CAFETERO": ["CALDAS", "RISARALDA", "QUINDIO"], "TOLIMA": ["TOLIMA"],
    "MONTERIA": ["CORDOBA"], "MOTERIA": ["CORDOBA"],
}


def _norm(serie):
    return (L.normalizar_texto(serie).str.replace(r"[^A-Z ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def cargar_divipola():
    dv = pd.read_csv(RAIZ / "datos" / "divipola.csv", dtype=str)
    dv["nom"] = _norm(dv["nom_mpio"])
    dv["dpto_norm"] = _norm(dv["dpto"])
    return dv


def departamentos_de_zona(zona):
    """Departamentos que menciona una zona comercial, ignorando los nombres de gerentes."""
    if not isinstance(zona, str):
        return set()
    salida = set()
    for segmento in zona.split("-"):
        partes = [p.strip() for p in segmento.split("/") if p.strip()]
        es_lista = "/" in segmento
        for p in partes:
            p = p.replace("ZONA ", "").strip()
            if p in REGIONES and (es_lista or p not in {"CESAR"}):  # "CESAR" suelto es el gerente
                salida.update(REGIONES[p])
    return salida


def candidatos_departamento(ciudad):
    """Códigos de departamento posibles para cada nombre de ciudad (más de uno si el nombre se repite en DIVIPOLA)."""
    dv = cargar_divipola()
    por_nombre = dv.groupby("nom")["cod_dpto"].agg(list)
    ciudad = _norm(pd.Series(ciudad)).replace(ALIAS)
    return [sorted({cod for n in VARIOS.get(c, [c]) if n in por_nombre.index for cod in por_nombre.loc[n]}) for c in ciudad]


def asignar_departamento(ciudad, zona):
    """Código DIVIPOLA del departamento por fila. NaN si la ciudad no existe o la ambigüedad no se puede resolver."""
    dv = cargar_divipola()
    por_nombre = dv.groupby("nom")[["cod_dpto", "dpto_norm"]].agg(list)
    ciudad = _norm(pd.Series(ciudad)).replace(ALIAS)
    zona = L.normalizar_texto(pd.Series(zona))
    salida = []
    for c, z in zip(ciudad, zona):
        nombres = VARIOS.get(c, [c])
        cands = [(cod, dp) for n in nombres if n in por_nombre.index
                 for cod, dp in zip(*por_nombre.loc[n])]
        cands = list(dict.fromkeys(cands))
        if len(cands) == 1:
            salida.append(cands[0][0])
            continue
        en_zona = [cod for cod, dp in cands if dp in departamentos_de_zona(z)]
        salida.append(en_zona[0] if len(set(en_zona)) == 1 else np.nan)
    return pd.Series(salida, index=getattr(ciudad, "index", None), dtype="object")
