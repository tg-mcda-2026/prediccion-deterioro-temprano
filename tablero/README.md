# Tablero de Power BI

`Tablero.pbix` es el archivo de Power BI Desktop. Sus tablas (carpeta `tablas/`) se calcularon a partir de
los resultados del proyecto, y el archivo guarda los datos, así que se abre con las cifras cargadas.
Todas las tablas son agregadas: ninguna tiene un registro por crédito. El parámetro **CarpetaTablas** trae la ruta
`C:\Tablero\tablas\`, la del equipo original; para volver a cargar los CSV en otro equipo, se cambia el
parámetro (Transformar datos > Editar parámetros) por la ruta de la carpeta `tablas/`, terminada en `\`, y se pulsa
**Actualizar**.

## Contenido de la carpeta

| Elemento | Qué es |
|---|---|
| `Tablero.pbix` | Archivo que se abre con Power BI Desktop: las cinco páginas, el modelo de datos (una tabla por CSV, con sus medidas) y los datos cargados |
| `Tablero.pdf` | El tablero exportado; se abre con cualquier lector de PDF |
| `tablas/` | Las tablas que alimentan el tablero (CSV con punto y coma y coma decimal) |

Páginas: 1. Población y deterioro · 2. Evolución por cosecha · 3. Desempeño del modelo · 4. Interpretabilidad ·
5. Uso del modelo e hipótesis.

## Tablas que generan el tablero

| Tabla | Filas | Columnas | Contenido u origen | Páginas |
|---|---|---|---|---|
| `tablas/anios.csv` | 3 | anio | años de desembolso (filtra la página 1) | 1 |
| `tablas/poblacion_anio.csv` | 3 | anio, creditos, con_resultado_6m, con_resultado_3m, con_resultado_60, con_resultado_90, eventos_mora30_6m, eventos_mora30_3m, eventos_mora60_6m, eventos_mora90_6m, pagos_anticipados | base de la población, agregada por año: créditos, créditos con resultado, eventos y pagos anticipados | 1, 2 |
| `tablas/poblacion_trimestre.csv` | 12 | anio, trimestre, creditos, con_resultado_6m, con_resultado_3m, eventos_mora30_6m, eventos_mora30_3m | base de la población, agregada por trimestre de desembolso | 1 |
| `tablas/por_independiente.csv` | 6 | anio, independiente, creditos, con_resultado_6m, eventos_mora30_6m, con_resultado_60, con_resultado_90, eventos_mora60_6m, eventos_mora90_6m | base de la población, agregada por año y tipo de independiente | 1 |
| `tablas/por_tipo_uso.csv` | 13 | anio, tipo_uso, creditos, con_resultado_6m, eventos_mora30_6m | base de la población, agregada por año y tipo de uso de la moto | 1 |
| `tablas/por_estado_civil.csv` | 16 | anio, estado_civil, creditos, con_resultado_6m, eventos_mora30_6m | base de la población, agregada por año y estado civil | 1 |
| `tablas/por_segmento.csv` | 9 | anio, segmento, creditos, con_resultado_6m, eventos_mora30_6m | base de la población, agregada por año y segmento comercial | 1 |
| `tablas/politica.csv` | 11 | trimestre, creditos, mora30_6m, mora30_3m, prenda, cuota_inicial, plazo_48_mas, tasa_mensual, aprobacion_automatica | protocolo/politica_cosechas.csv (trimestres con 50 desembolsos o más) | 2 |
| `tablas/tasa_regla.csv` | 3 | anio, regla, orden, tasa, creditos | protocolo/tasa_por_regla.csv (mora 30+ a 6 meses por año de desembolso) | 2 |
| `tablas/versiones.csv` | 6 | version, variables, auc_cv, diferencia, ic_inf, ic_sup, elegida | protocolo/decision_tabla.csv | 3 |
| `tablas/objetivos.csv` | 8 | evento, orden, modelo, auc, ic_inf, ic_sup, creditos, eventos | protocolo/reporte_2025/monitoreo_confirmacion.csv | 3 |
| `tablas/algoritmos.csv` | 8 | algoritmo, auc_cv, auc_2025, pr_auc, ks, brier, pendiente | protocolo/c3_detalle/metricas_2025.csv | 3 |
| `tablas/umbral.csv` | 3 | marcados, sensibilidad, especificidad, precision, f1, deterioro_no_marcados | protocolo/c3_detalle/metricas_umbral_2025.csv | 3 |
| `tablas/calibracion.csv` | 5 | periodo, orden, observada, predicha, observado_esperado | hipotesis/calibracion/calibracion_en_el_conjunto.csv | 3 |
| `tablas/deciles.csv` | 10 | decil, creditos, tasa, prob_media | protocolo/reporte_2025/deciles_vs_segmento.csv | 3 |
| `tablas/importancia.csv` | 6 | variable, contribucion_pp, participacion, caida_auc | protocolo/c3_detalle/shap_global.csv e importancia_permutacion_2025.csv | 4 |
| `tablas/contribucion.csv` | 36 | variable, categoria, creditos, contribucion_pp, deterioro_observado | protocolo/c3_detalle/shap_por_categoria.csv (categorías con diez créditos o más) | 4 |
| `tablas/frontera.csv` | 51 | aprobado, det_modelo, det_segmento, cap_modelo, cap_segmento | hipotesis/h2/frontera.csv | 5 |
| `tablas/reordenacion.csv` | 12 | aprobacion, metrica, modelo, segmento, diferencia, ic_inf, ic_sup | hipotesis/h2/reordenacion.csv | 5 |
| `tablas/composicion.csv` | 28 | atributo, categoria, creditos, no_aprobacion, deterioro | hipotesis/h2/composicion.csv (corte del 80 %, categorías con veinte créditos o más) | 5 |
| `tablas/composicion_resumen.csv` | 3 | aprobacion, aprobados, no_aprobados, deterioro_aprobados, deterioro_no_aprobados, eventos_no_aprobados, pct_eventos_evitados, capital_medio_aprobados, capital_medio_no_aprobados, pct_capital_no_aprobado | hipotesis/h2/composicion_resumen.csv | 5 |
| `tablas/equidad.csv` | 6 | grupo, atributo, creditos, eventos, mora, pct_marcados, verdaderos_positivos, falsos_positivos, auc, observado_esperado | hipotesis/h3/grupos_2025.csv (marcando el 20 %) | 5 |
| `tablas/bloques.csv` | 5 | bloque, auc_2025, ic_inf, ic_sup | hipotesis/h1/prueba_2025_bloques.csv y protocolo/reporte_2025/discriminacion_30_6m.csv | 5 |

Como en el documento, los créditos que salen del panel sin deterioro son pagos anticipados y cuentan como buenos
pagadores.
