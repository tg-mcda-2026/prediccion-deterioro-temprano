"""
Limpieza y delimitación de la población (se documenta paso a paso en notebooks/).

Población: primer crédito de cada cliente, que sea de motocicleta (sin factoring) y de un
cliente independiente (formal o informal), con las variables tomadas del corte de desembolso (C-0).
Se descarta el corte reextraído (enero 2026), que reescribe atributos del cliente y del contrato.

Objetivo (mora 30+/60+/90+ en los primeros H meses):
  - la obligación completa H meses en el panel      -> mora observada
  - sale del panel antes de H con mora >= umbral    -> malo
  - sale del panel antes de H sin llegar al umbral  -> vacío (NaN) en esta base: es un pago anticipado y
                                                       protocolo.agregar_etiquetas lo cuenta como buen pagador
  - no han pasado H meses al último corte           -> desconocido (NaN)

Salidas (carpeta datos/):
  - base_modelo.parquet        una fila por obligación, ordenada por fecha de desembolso
  - diccionario_variables.csv  variable, escenario / motivo de exclusión
"""
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "datos" / "cartera.parquet"
SALIDA = RAIZ / "datos"
SALIDA.mkdir(exist_ok=True)

SECTORES_INDEPENDIENTES = {"formal_independent", "informal_independent"}
HORIZONTES = (3, 6, 9, 12)  # meses después del desembolso (MOB)
VALORES_FALTANTES = {"", "NONE", "NAN", "NULL", "SIN DATOS", "SIN DATO", "?", "-", "'", "N/A", "NA"}


def fecha_excel(serie):
    return pd.to_datetime(serie, unit="D", origin="1899-12-30")


def normalizar_texto(serie):
    """Mayúsculas, sin tildes, sin apóstrofo de Excel, espacios colapsados; faltantes a NaN."""
    def limpiar(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        v = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode("ascii")
        v = re.sub(r"\s+", " ", v.replace("'", "")).strip().upper()
        return np.nan if v in VALORES_FALTANTES else v
    return serie.map(limpiar)


def codigo_entero(serie):
    """'0141', 141 y '141' son el mismo código."""
    return pd.to_numeric(normalizar_texto(serie), errors="coerce").astype("Int64")


# --------------------------------------------------------------------------- producto
# Familia comercial de la moto (TIPO_MOTO trae el mismo modelo escrito de varias formas) y tipo de uso. La tabla que
# traduce cada escritura a su familia y a su tipo de uso tiene los nombres de los modelos, que identifican a la entidad:
# por confidencialidad está en datos/familias_moto.csv (patrón, familia y tipo de uso, en el orden en que se buscan) y
# no se entrega, como la base.
TABLA_FAMILIAS = RAIZ / "datos" / "familias_moto.csv"
if TABLA_FAMILIAS.exists():
    _tabla = pd.read_csv(TABLA_FAMILIAS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    FAMILIAS_MOTO = list(zip(_tabla["patron"], _tabla["familia"]))
    TIPO_USO = dict(zip(_tabla["familia"], _tabla["tipo_uso"]))  # tipo de moto por uso (más estable que la familia)
else:
    FAMILIAS_MOTO, TIPO_USO = [], {}

# Por confidencialidad, al mostrar resultados cada familia se identifica con su tipo de uso y una letra según su
# frecuencia en la base (A = la más frecuente). No cambia la variable del modelo.
NOMBRE_USO = {"SCOOTER": "Scooter", "DOBLE_PROPOSITO": "Doble propósito", "NAKED_SPORT": "Deportiva",
              "URBANA_TRABAJO": "Urbana y de trabajo", "ATV": "Cuatrimoto"}


def etiquetas_familia(base):
    """{familia_moto: etiqueta genérica} a partir de una base con familia_moto y tipo_uso_moto."""
    n = (base.groupby(["tipo_uso_moto", "familia_moto"]).size().rename("n").reset_index()
         .sort_values(["tipo_uso_moto", "n", "familia_moto"], ascending=[True, False, True]))
    letra = n.groupby("tipo_uso_moto").cumcount().map(lambda k: chr(ord("A") + k))
    return dict(zip(n.familia_moto, n.tipo_uso_moto.map(NOMBRE_USO).fillna("Otra") + " " + letra))


def familia_moto(tipo):
    if not FAMILIAS_MOTO:
        raise FileNotFoundError(f"falta {TABLA_FAMILIAS}: la tabla de familias de moto no se entrega con el código")
    if pd.isna(tipo):
        return np.nan
    for patron, familia in FAMILIAS_MOTO:
        if re.search(patron, tipo):
            return familia
    return "OTRA"


# --------------------------------------------------------------------------- variables
# Escenario de cada variable del modelo. Segment/Approval quedan solo como referencia.
ESCENARIOS = {
    "sexo": "sociodemografico",
    "edad": "sociodemografico",
    "estado_civil": "sociodemografico",
    "estrato": "sociodemografico",
    "personas_a_cargo": "sociodemografico",
    "nivel_estudio": "sociodemografico",
    "independiente_informal": "sociodemografico",
    "ciudad": "sociodemografico",
    "familia_moto": "producto",
    "tipo_uso_moto": "producto",
    "cilindraje": "producto",
    "precio_moto": "producto",
    "propietario_moto_cliente": "producto",
    "zona_homologada": "producto",
    "codigo_distribuidor": "producto",
    "mes_desembolso": "producto",
    "indice_mora_ciudad": "historica (entorno de la solicitud)",
    "indice_mora_tienda": "historica (entorno de la solicitud)",
    "tipo_ciudad": "derivada",
    "mujer_scooter": "derivada",
    "joven_25": "derivada",
    "ref_segmento": "referencia (no entra al modelo)",
    "ref_aprobacion": "referencia (no entra al modelo)",
}

EXCLUIDAS = {
    "Wage / Rang_Wage": "salario: no verificable en independientes",
    "CUOTA_INICIAL / %Initial_fee / Initial_fee": "condición del crédito: refleja capacidad de pago",
    "Time_limit / Rang_TimeLimit": "plazo: condición del crédito",
    "Interest_Rate": "tasa: condición del crédito",
    "Valor Capital / Valor Cuota / Linea de credito": "monto y producto financiero",
    "Seguro Vida / Seguro todo riesgo": "se calculan sobre el crédito",
    "Prenda / Codeudor": "garantías del crédito (Prenda excluida por solicitud)",
    "Segment": "calificación interna: solo para comparar",
    "Approval / Usuario Aprueba / Usuario Crea": "resultado del proceso de aprobación",
    "Nombre_Empresa": "95% 'Sin datos' en independientes",
    "PROFESSION / Productos / Tipo Identificacion": "casi vacía o constante en la población",
    "Nombre de la tienda / Nombre de la tienda Actual / Codigo Vendedor": "alta cardinalidad; se usa distribuidor y zona",
    "Zona / Nombre Zona (originales)": "el código cambia de significado entre años (reorganización comercial); "
                                       "se reemplaza por zona_homologada (zona 2025 de la ciudad)",
    "Dias Mora, saldos, intereses, flags, Cobro Juridico, Cuota cero": "desempeño posterior al desembolso (solo para construir el objetivo)",
    "Ultimo Pago / Mentorias / Otros Cobros / Marca_Cosecha": "vacías, constantes o duplicadas",
}


def meses_entre(desde, hasta):
    return (hasta.dt.year - desde.dt.year) * 12 + (hasta.dt.month - desde.dt.month)


def construir_objetivos(panel, ultimo_corte):
    """Objetivos de mora por horizonte con la regla de salidas descrita en el encabezado."""
    panel = panel.sort_values(["Numero_Encriptado", "Corte"])
    g = panel.groupby("Numero_Encriptado")
    info = pd.DataFrame({
        "mob_max": g["Mes_Cosecha"].max(),
        "sale_del_panel": g["Corte"].max() < ultimo_corte,
        "mora_final": g["Dias Mora"].last(),
    })
    info["meses_disponibles"] = meses_entre(fecha_excel(g["Periodo_Des"].first()),
                                            pd.Series(fecha_excel(ultimo_corte), index=info.index))
    objetivos = [info]
    for h in HORIZONTES:
        madura = info["meses_disponibles"] >= h
        sale_antes = info["sale_del_panel"] & (info["mob_max"] < h)
        ventana = panel[panel["Mes_Cosecha"].between(1, h)].groupby("Numero_Encriptado")
        for umbral in (30, 60, 90):
            malo = ventana[f"Flag_Over_{umbral}"].max().reindex(info.index).fillna(0)
            y = malo.where(info["mob_max"] >= h)
            y[sale_antes & (malo == 1)] = 1
            objetivos.append(y.where(madura).rename(f"mora{umbral}_{h}m"))
        estado = np.select(
            [~madura, ~sale_antes, info["mob_max"] == 0, info["mora_final"] == 0, info["mora_final"] >= 30],
            ["inmaduro", "observado", "sale_sin_exposicion", "sale_al_dia", "sale_con_mora"], "sale_mora_menor_30")
        objetivos.append(pd.Series(estado, index=info.index, name=f"estado_{h}m"))
    # Primer pago: no pagó la primera cuota (MOB 1)
    mob1 = panel[panel["Mes_Cosecha"] == 1].set_index("Numero_Encriptado")
    objetivos.append((mob1["Cuota cero"] == "No pago").astype(float).rename("no_pago_primera_cuota"))
    return pd.concat(objetivos, axis=1)


def zona_homologada(df_c0):
    """Zona de cada ciudad según la estructura comercial de 2025 (los códigos cambian entre años)."""
    d = df_c0[fecha_excel(df_c0["Corte"]).dt.year == 2025].copy()
    d["ciudad"] = normalizar_texto(d["Ciudad"])
    d["zona"] = codigo_entero(d["Zona"])
    nombres = normalizar_texto(d.groupby("zona")["Nombre Zona"].agg(lambda s: s.mode().iloc[0]))
    validas = nombres[~nombres.str.startswith("SIN ASIGNAR")]
    d = d[d["zona"].isin(validas.index)]
    mapa = d.groupby("ciudad")["zona"].agg(lambda s: s.mode().iloc[0])
    etiquetas = {z: f"Z{z:02d} {n.split('-')[0].strip()}" for z, n in validas.items()}
    return mapa.map(etiquetas)


def filtrar_poblacion(df):
    """Embudo de población. Devuelve el panel filtrado, el registro C-0 por obligación, el embudo y el último corte."""
    embudo = []

    def paso(nombre, d, motivo=""):
        embudo.append({"paso": nombre, "filas": len(d), "obligaciones": d["Numero_Encriptado"].nunique(),
                       "clientes": d["Cedula_Encriptado"].nunique(), "motivo": motivo})
        return d

    paso("Base cruda", df, "una fila por obligación y mes de corte")
    df = paso("Sin el corte reextraído (ene-2026)", df[df["Corte"] < df["Corte"].max()],
              "reescribe atributos del cliente y del contrato")
    ultimo_corte = df["Corte"].max()

    # Primer crédito de cada cliente en toda la cartera: sin historial interno previo.
    # Varios créditos el mismo día: el menor identificador, para que sea reproducible.
    primeros = (df[["Cedula_Encriptado", "Numero_Encriptado", "Fecha desembolso"]].drop_duplicates()
                .sort_values(["Cedula_Encriptado", "Fecha desembolso", "Numero_Encriptado"])
                .drop_duplicates("Cedula_Encriptado")["Numero_Encriptado"])
    df = paso("Primer crédito de cada cliente", df[df["Numero_Encriptado"].isin(primeros)],
              "los créditos siguientes ya tienen historial de pago con la entidad")
    df = paso("Sin factoring", df[df["Tipo"] == "01CR"], "línea para distribuidores (NIT), no personas")
    df = paso("Solo motocicletas", df[df["Productos"] == "Motocicletas"], "instrumentos musicales: otra línea de negocio")

    # Atributos al desembolso: el sistema los sobrescribe en cortes posteriores.
    c0 = df[df["Mes_Cosecha"] == 0].drop_duplicates("Numero_Encriptado").set_index("Numero_Encriptado")
    independientes = c0.index[c0["SECTOR_ECONOMICO"].isin(SECTORES_INDEPENDIENTES)]
    df = paso("Independientes (formal o informal) al desembolso", df[df["Numero_Encriptado"].isin(independientes)],
              "sector económico en el mes de desembolso")
    return df, c0.loc[independientes], embudo, ultimo_corte


def construir_variables(c0, mapa_zonas):
    """Variables normalizadas, tomadas del registro C-0."""
    base = pd.DataFrame(index=c0.index)
    base["id_cliente"] = c0["Cedula_Encriptado"]
    base["fecha_desembolso"] = fecha_excel(c0["Fecha desembolso"])
    base["cosecha"] = base["fecha_desembolso"].dt.to_period("M").astype(str)
    base["mes_desembolso"] = base["fecha_desembolso"].dt.month

    # Sociodemográficas
    base["sexo"] = normalizar_texto(c0["Sexo"])
    base["edad"] = c0["Age"]
    base["estado_civil"] = normalizar_texto(c0["Estado Civil"]).str.replace(r"\(A\)", "", regex=True).str.strip()
    estrato = pd.to_numeric(normalizar_texto(c0["Estrato"]), errors="coerce")
    # El estrato 7 no existe en Colombia; según la entidad es un error de digitación y se reemplaza por la moda de los
    # demás, calculada con los desembolsos anteriores a 2025 para no usar la cohorte de prueba
    validos = estrato.where(estrato.between(1, 6))
    moda = validos[base["fecha_desembolso"] < pd.Timestamp("2025-01-01")].mode().iloc[0]
    base["estrato"] = validos.mask(estrato == 7, moda)
    cargo = pd.to_numeric(normalizar_texto(c0["Personas A Cargo"]), errors="coerce")
    base["personas_a_cargo"] = cargo.where(cargo <= 8)  # 10 y 14: digitación
    base["nivel_estudio"] = normalizar_texto(c0["Nivel Estudio"]).replace({"MAESTRIA": "POSTGRADO"})
    base["independiente_informal"] = (c0["SECTOR_ECONOMICO"] == "informal_independent").astype(int)
    base["ciudad"] = normalizar_texto(c0["Ciudad"])

    # Producto y canal
    tipo = normalizar_texto(c0["TIPO_MOTO"])
    base["modelo_moto_original"] = tipo
    base["familia_moto"] = tipo.map(familia_moto)
    base["tipo_uso_moto"] = base["familia_moto"].map(TIPO_USO).fillna("OTRA")
    cil = pd.to_numeric(normalizar_texto(c0["Cilindraje"]), errors="coerce")
    moda = cil.groupby(base["familia_moto"]).agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    base["cilindraje_imputado"] = cil.isna().astype(int)
    base["cilindraje"] = cil.fillna(base["familia_moto"].map(moda))
    base["precio_moto"] = c0["Product_Price"]
    base["propietario_moto_cliente"] = (normalizar_texto(c0["Propietario Moto"]) == "CLIENTE").astype(int)
    base["zona_original"] = codigo_entero(c0["Zona"])  # solo para auditoría; no entra al modelo
    base["zona_homologada"] = base["ciudad"].map(mapa_zonas)
    base["codigo_distribuidor"] = codigo_entero(c0["Codigo del distribuidor"])

    # Referencia (no entra al modelo)
    base["ref_segmento"] = normalizar_texto(c0["Segment"])
    base["ref_aprobacion"] = normalizar_texto(c0["Approval"])
    return base


# --------------------------------------------------------------------------- variables históricas
# Mora del entorno donde vive y compra el cliente (ciudad y tienda), calculada en la fecha de la solicitud solo con
# créditos de moto ANTERIORES cuyo resultado a 6 meses ya se conocía. No usa datos del cliente ni del vendedor
# (decisión de negocio: la experiencia del vendedor no debe afectar el crédito del cliente). Distribuidor y su
# volumen se descartaron: el distribuidor repite a la tienda (correlación 0,90) y el volumen no mostró señal.
#
# Se expresa como ÍNDICE RELATIVO: mora del entorno / mora general de la cartera en esa misma fecha (1,3 = 30% más
# mora que el promedio). Así no arrastra el nivel de la época (la mora conocida llega con ~7 meses de rezago).
# Sin historia suficiente el índice vale 1,0 (neutro); no se agrega un indicador de "sin historia" porque casi
# todos esos casos son de 2023 y funcionaría como un marcador del año.
MESES_PARA_CONOCER_RESULTADO = 7  # desembolso + 6 meses de ventana + 1 mes de corte
SUAVIZADO = 20                    # créditos "ficticios" con la mora general: poca historia -> cerca de 1,0
MINIMO_CARTERA = 500              # créditos conocidos en toda la cartera para que la mora general sea confiable


def variables_historicas(df, base, c0, ultimo_corte):
    serie = df[(df["Corte"] < df["Corte"].max()) & (df["Tipo"] == "01CR") & (df["Productos"] == "Motocicletas")]
    todos_c0 = serie[serie["Mes_Cosecha"] == 0].drop_duplicates("Numero_Encriptado").set_index("Numero_Encriptado")
    hist = pd.DataFrame({
        "desembolso": fecha_excel(todos_c0["Fecha desembolso"]),
        "ciudad": normalizar_texto(todos_c0["Ciudad"]),
        "tienda": pd.to_numeric(todos_c0["Codigo de la tienda"], errors="coerce"),
    })
    hist["malo"] = construir_objetivos(serie, ultimo_corte)["mora30_6m"].reindex(hist.index)
    hist["conocido_desde"] = hist["desembolso"] + pd.DateOffset(months=MESES_PARA_CONOCER_RESULTADO)
    conocidos = hist[hist["malo"].notna()].sort_values("conocido_desde")

    fechas = base["fecha_desembolso"].values
    idx = np.searchsorted(conocidos["conocido_desde"].values, fechas, side="right")
    acum = np.concatenate([[0], conocidos["malo"].cumsum().values])
    mora_general = np.where(idx >= MINIMO_CARTERA, acum[idx] / np.maximum(idx, 1), np.nan)

    claves = {"ciudad": base["ciudad"].values,
              "tienda": pd.to_numeric(c0.loc[base.index, "Codigo de la tienda"], errors="coerce").values}
    salida = pd.DataFrame(index=base.index)
    for nombre, clave in claves.items():
        n_prev, malos_prev = np.zeros(len(base)), np.zeros(len(base))
        for valor, grupo in conocidos.groupby(nombre):
            en = np.where(clave == valor)[0]
            if len(en) == 0:
                continue
            k = np.searchsorted(grupo["conocido_desde"].values, fechas[en], side="right")
            acum_g = np.concatenate([[0], grupo["malo"].cumsum().values])
            n_prev[en], malos_prev[en] = k, acum_g[k]
        suavizada = (malos_prev + SUAVIZADO * mora_general) / (n_prev + SUAVIZADO)
        salida[f"indice_mora_{nombre}"] = np.where(np.isnan(mora_general), 1.0, suavizada / mora_general)
        salida[f"creditos_hist_{nombre}"] = np.where(np.isnan(mora_general), 0, n_prev)  # diagnóstico, no entra al modelo
    return salida


# --------------------------------------------------------------------------- variables derivadas
# Solo a partir de variables permitidas (sociodemográficas y producto). De 8 candidatas, estas 3 mostraron señal
# dentro de cada año (AUC intra-año 0,52-0,54); las demás (soltero joven, deportiva joven, alta cilindrada joven,
# canal directo de la entidad, incoherencias de la solicitud) quedaron en ~0,50.
CAPITALES = {"BOGOTA", "MEDELLIN", "CALI", "BARRANQUILLA", "CARTAGENA", "BUCARAMANGA", "CUCUTA", "PEREIRA",
             "MANIZALES", "IBAGUE", "VILLAVICENCIO", "PASTO", "MONTERIA", "NEIVA", "ARMENIA", "POPAYAN",
             "VALLEDUPAR", "SANTA MARTA", "SINCELEJO", "TUNJA", "QUIBDO", "RIOHACHA", "FLORENCIA", "YOPAL",
             "ARAUCA", "MOCOA", "SAN JOSE DEL GUAVIARE", "LETICIA", "SAN ANDRES", "PUERTO CARRENO", "INIRIDA", "MITU"}
AREAS_METROPOLITANAS = {"BELLO", "ITAGUI", "ENVIGADO", "SABANETA", "LA ESTRELLA", "CALDAS", "COPACABANA",
                        "GIRARDOTA", "BARBOSA", "SOACHA", "CHIA", "MOSQUERA", "FUNZA", "MADRID", "COTA", "CAJICA",
                        "ZIPAQUIRA", "FACATATIVA", "SOLEDAD", "MALAMBO", "PUERTO COLOMBIA", "GALAPA",
                        "FLORIDABLANCA", "GIRON", "PIEDECUESTA", "VILLA DEL ROSARIO", "LOS PATIOS", "DOSQUEBRADAS",
                        "LA VIRGINIA", "JAMUNDI", "YUMBO", "PALMIRA", "CANDELARIA"}


def variables_derivadas(base):
    ciudad = base["ciudad"].fillna("")
    salida = pd.DataFrame(index=base.index)
    salida["tipo_ciudad"] = np.select([ciudad.isin(CAPITALES), ciudad.isin(AREAS_METROPOLITANAS)],
                                      ["CAPITAL", "AREA_METROPOLITANA"], "MUNICIPIO_INTERMEDIO")
    salida.loc[base["ciudad"].isna(), "tipo_ciudad"] = np.nan
    salida["mujer_scooter"] = ((base["sexo"] == "F") & (base["tipo_uso_moto"] == "SCOOTER")).astype(int)
    salida["joven_25"] = (base["edad"] < 25).astype(int)
    return salida


def construir_base(df):
    """Población + variables + objetivos. Devuelve la base del modelo y el embudo completo."""
    todas_c0 = df[(df["Tipo"] == "01CR") & (df["Productos"] == "Motocicletas") & (df["Mes_Cosecha"] == 0)]
    mapa_zonas = zona_homologada(todas_c0)
    panel, c0, embudo, ultimo_corte = filtrar_poblacion(df)

    base = construir_variables(c0, mapa_zonas).join(construir_objetivos(panel, ultimo_corte))
    base = base.join(variables_historicas(df, base, c0, ultimo_corte))
    base = base.join(variables_derivadas(base))
    base = base.sort_values(["fecha_desembolso", "id_cliente"]).rename_axis("id_obligacion").reset_index()

    madura = base[base["estado_6m"] != "inmaduro"]
    embudo.append({"paso": "Con 6 meses cumplidos al último corte", "obligaciones": len(madura),
                   "clientes": madura["id_cliente"].nunique(), "motivo": "créditos recientes: aún no cumplen 6 meses"})
    embudo = pd.DataFrame(embudo)
    embudo["salen"] = embudo["clientes"].shift() - embudo["clientes"]
    return base, embudo, fecha_excel(pd.Series([ultimo_corte]))[0]


def main():
    base, embudo, ultimo_corte = construir_base(pd.read_parquet(ENTRADA))
    base.to_parquet(SALIDA / "base_modelo.parquet", index=False)
    embudo.to_csv(SALIDA / "embudo_poblacion.csv", index=False, encoding="utf-8-sig")

    dicc = [(v, e, "") for v, e in ESCENARIOS.items()] + [(v, "excluida", m) for v, m in EXCLUIDAS.items()]
    pd.DataFrame(dicc, columns=["variable", "escenario", "motivo_exclusion"]).to_csv(
        SALIDA / "diccionario_variables.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ reporte
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 70)
    print("\n== Embudo de población ==")
    print(embudo.to_string(index=False))
    print(f"\nDesembolsos: {base.fecha_desembolso.min():%Y-%m-%d} a {base.fecha_desembolso.max():%Y-%m-%d}"
          f" | último corte válido: {ultimo_corte:%Y-%m-%d}")
    print("\n== Estado a 6 meses ==")
    print(base["estado_6m"].value_counts().to_string())
    print("\n== Objetivos: n con resultado y tasa de malos ==")
    obj = [c for c in base.columns if (c.startswith("mora") and c != "mora_final") or c == "no_pago_primera_cuota"]
    print(pd.DataFrame({"n": base[obj].notna().sum(), "tasa_malo": base[obj].mean().round(3)}).to_string())
    print("\n== Tasa de malos a 6 meses por año de desembolso ==")
    print(base.groupby(base.fecha_desembolso.dt.year)[["mora30_6m", "mora60_6m", "mora90_6m"]]
          .agg(["count", "mean"]).round(3).to_string())
    print("\n== Zona homologada por año de desembolso ==")
    print(pd.crosstab(base.zona_homologada.fillna("SIN ZONA"), base.fecha_desembolso.dt.year, normalize="columns").round(2).to_string())
    print("\n== Nulos en variables del modelo ==")
    vars_modelo = [v for v, e in ESCENARIOS.items() if not e.startswith("ref")]
    print(base[vars_modelo].isna().sum()[lambda s: s > 0].to_string())
    sin_mapear = base.loc[base.familia_moto == "OTRA", "modelo_moto_original"].unique()
    print("\nmodelos de moto sin familia:", list(sin_mapear))
    print(f"ciudades distintas tras normalizar: {base.ciudad.nunique()}")


if __name__ == "__main__":
    main()
