# Predicción del deterioro temprano de pago en clientes independientes de financiación de motocicletas

Trabajo de grado (caso de estudio) · Maestría en Ciencias de los Datos y la Analítica · Universidad EAFIT

Material complementario del trabajo de grado, descrito en su Anexo A: los notebooks del análisis, los scripts que usan y el tablero de resultados.

## Contenido

- **[notebooks/](notebooks)**: los ocho notebooks del análisis (`.ipynb`), ejecutados con sus salidas reales. GitHub los muestra en el navegador; también se abren con Jupyter. Las cifras, las tablas y las figuras del documento proceden de esas salidas.
- **[src/](src)**: los scripts que usan los notebooks, sin datos. Sin la base no se pueden ejecutar.
- **[tablero/](tablero)**: tablero de Power BI con los resultados principales, en cinco páginas:
  - `Tablero.pdf`: el tablero exportado; se abre con cualquier lector de PDF;
  - `Tablero.pbix`: el archivo de Power BI Desktop, que ya trae los datos cargados.

  Todas sus tablas son agregadas (`tablas/`): ninguna tiene un registro por crédito. [tablero/README.md](tablero/README.md) describe cada tabla.

## Notebooks

| N.º | Notebook | Contenido |
|---|---|---|
| 01 | [Limpieza y población](notebooks/01_limpieza_poblacion.ipynb) | Controles de integridad, sobrescritura de atributos, corte reextraído, zona, embudo, objetivo y pagos anticipados. |
| 02 | [Procesamiento de variables](notebooks/02_procesamiento_variables.ipynb) | Normalización de cada variable, variables del lugar de compra y política de variables. |
| 03 | [Selección de variables](notebooks/03_seleccion_variables.ipynb) | Revisión univariada y eliminación hacia atrás, solo con 2023-2024. |
| 04 | [Versiones y decisión](notebooks/04_versiones_y_decision.ipynb) | Versiones candidatas, regla de decisión y costo de excluir el sexo. |
| 05 | [Prueba 2025](notebooks/05_prueba_2025.ipynb) | Discriminación, calibración, monitoreo, equidad por sexo y deciles frente al segmento. |
| 06 | [Estabilidad y diagnósticos](notebooks/06_estabilidad_y_diagnosticos.ipynb) | Población con resultado, caída entre cosechas y condiciones de aprobación, cálculo de los días de mora y señales de cobranza, efecto de cosecha, sobreajuste, deriva, curva de aprendizaje, PSI y prueba de fuga. |
| 07 | [Contraste de las hipótesis](notebooks/07_contraste_hipotesis.ipynb) | Bloques de información, reordenación y composición de la cartera, equidad por sexo y estrato y calibración en el conjunto. |
| 08 | [Métricas e interpretabilidad](notebooks/08_metricas_e_interpretabilidad.ipynb) | Métricas completas de C3, algoritmos por separado y valores de Shapley. |

## Tablero

1. Población y deterioro
2. Evolución por cosecha
3. Desempeño del modelo
4. Interpretabilidad
5. Uso del modelo e hipótesis

## Scripts (src/)

Cada script lleva el número del notebook que muestra sus resultados.

| Notebook | Scripts |
|---|---|
| 01 y 02 | [limpieza.py](src/limpieza.py) y [divipola.py](src/divipola.py) |
| 03 | [03a_revision_variables.py](src/03a_revision_variables.py) y [03b_seleccion_variables.py](src/03b_seleccion_variables.py) |
| 04 | [04a_versiones_cv.py](src/04a_versiones_cv.py) y [04b_decision.py](src/04b_decision.py) |
| 05 | [05_prueba_2025.py](src/05_prueba_2025.py) |
| 06 | [06a_diagnosticos.py](src/06a_diagnosticos.py), [06b_politica_cosechas.py](src/06b_politica_cosechas.py) y [06c_mora_cobranza.py](src/06c_mora_cobranza.py) |
| 07 | [07a_hipotesis1_bloques.py](src/07a_hipotesis1_bloques.py) y [07b_hipotesis2_3.py](src/07b_hipotesis2_3.py) |
| 08 | [08_metricas_interpretabilidad.py](src/08_metricas_interpretabilidad.py) |

Módulos que comparten: [protocolo.py](src/protocolo.py) (reglas del protocolo y sellado de 2025), [modelos.py](src/modelos.py) (los siete algoritmos), [credito.py](src/credito.py) (condiciones del crédito) y [utilidades_auc.py](src/utilidades_auc.py) (pruebas estadísticas).

## Confidencialidad: qué no se incluye

Los datos son de la entidad que los aportó y están protegidos por un acuerdo de confidencialidad. Por eso este repositorio no incluye la base de cartera, ni las bases derivadas de ella, ni archivos con un registro por crédito, ni el nombre de la entidad. Tampoco incluye la tabla que agrupa los modelos de moto en familias (`datos/familias_moto.csv`), cuyos nombres identifican a la entidad: en las salidas, cada familia aparece con su tipo de uso y una letra (por ejemplo, «Scooter A»). Sin la base, los notebooks no se pueden volver a ejecutar; por eso se publican ya ejecutados, con las salidas obtenidas con ella. Tampoco se incluyen los archivos de resultados que leen los notebooks: su contenido ya está impreso en las salidas.
