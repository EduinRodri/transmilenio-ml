# Pruebas realizadas

- **Fecha de ejecución:** 6 de octubre de 2026
- **Entorno:** Windows 11 Pro · Python 3.13.15 · pandas 3.0.6 · scikit-learn 1.8.0 · scipy 1.18.1 ·
  matplotlib 3.11.2 · pytest 9.1.1
- **Comando:** `uv run pytest -v`
- **Resultado:** **33 de 33 pruebas aprobadas** (69 s)

Las pruebas están en la carpeta [`pruebas/`](../pruebas) y cubren tres niveles:

1. **Unitarias con valores calculables a mano.** Verifican que las fórmulas estén bien escritas,
   por ejemplo que la entropía de un grupo mitad y mitad sea exactamente 1 bit.
2. **De integridad de datos.** Verifican que la limpieza no pierda ni invente información.
3. **De comportamiento del modelo.** Verifican que los modelos aprendan algo útil, sean
   reproducibles y den resultados con sentido en el mundo real.

## 1. Carga y limpieza de datos — `test_datos.py` (13 pruebas)

| # | Prueba | Qué verifica | Resultado |
|---|---|---|---|
| 1 | `test_el_crudo_no_trae_filas_vacias` | Se eliminan las 46.133 filas vacías y las columnas sin nombre del archivo original | Aprobada |
| 2 | `test_el_crudo_trae_los_31_dias_de_octubre` | El archivo tiene una columna por cada día del mes | Aprobada |
| 3 | `test_quitar_codigo` | `(06000) Portal Eldorado` pasa a `Portal Eldorado`; un nombre sin código no cambia | Aprobada |
| 4-8 | `test_tipo_de_dia` (5 casos) | 1 oct → hábil · 5 oct → sábado · 6 oct → domingo · **14 oct → festivo** · 15 oct → hábil | Aprobadas |
| 9 | `test_el_formato_largo_no_pierde_salidas` | Pasar de formato ancho a largo conserva el total (más de 58 millones de salidas) | Aprobada |
| 10 | `test_no_hay_valores_faltantes_ni_negativos` | Ninguna celda vacía y ningún conteo negativo | Aprobada |
| 11 | `test_se_descartan_los_puntos_que_no_son_estaciones` | Todas las estaciones que quedan tienen al menos 100 salidas en un día hábil; el bicicletero ya no está | Aprobada |
| 12 | `test_los_portales_se_marcan_incluidas_las_cabeceras` | Portal Américas y Cabecera Autopista Norte cuentan como portal; Calle 57 no | Aprobada |
| 13 | `test_un_dia_habil_mueve_mas_gente_que_un_domingo` | Prueba de cordura: hábil > sábado > domingo o festivo | Aprobada |

## 2. Árbol de decisión — `test_supervisado.py` (11 pruebas)

| # | Prueba | Qué verifica | Resultado |
|---|---|---|---|
| 14 | `test_entropia_de_un_grupo_puro_es_cero` | H(alta, alta, alta) = 0 | Aprobada |
| 15 | `test_entropia_de_dos_clases_mitad_y_mitad_es_un_bit` | H(alta, baja, alta, baja) = 1 bit | Aprobada |
| 16 | `test_entropia_maxima_con_tres_clases` | H(baja, media, alta) = log₂3 ≈ 1,585 bits | Aprobada |
| 17 | `test_ganancia_de_una_pregunta_que_separa_perfecto` | Una pregunta que separa las clases del todo gana 1 bit | Aprobada |
| 18 | `test_ganancia_de_una_pregunta_inutil_es_cero` | Una pregunta que no separa nada gana 0 bits | Aprobada |
| 19 | `test_umbrales_y_etiquetas` | Los terciles generan las tres clases; el valor más bajo es "baja" y el más alto "alta" | Aprobada |
| 20 | `test_los_conjuntos_no_se_mezclan` | Entrenamiento, validación y prueba no comparten ningún ejemplo | Aprobada |
| 21 | `test_el_entrenamiento_tiene_las_tres_clases_balanceadas` | Cada clase es entre el 30 % y el 37 % del entrenamiento | Aprobada |
| 22 | `test_elegir_profundidad_prefiere_el_arbol_mas_simple` | Si dos profundidades aciertan casi igual, se elige la menor | Aprobada |
| 23 | `test_el_arbol_supera_la_linea_base` | El árbol acierta al menos 30 puntos más que responder siempre la clase más común | Aprobada |
| 24 | `test_el_arbol_es_reproducible` | Dos entrenamientos iguales dan las mismas predicciones | Aprobada |

## 3. Agrupamiento — `test_no_supervisado.py` (9 pruebas)

| # | Prueba | Qué verifica | Resultado |
|---|---|---|---|
| 25 | `test_distancia_euclidiana_a_mano` | d((0,0), (3,4)) = 5; la distancia de un punto a sí mismo es 0 | Aprobada |
| 26 | `test_cada_perfil_suma_uno` | Cada perfil horario es una distribución: sus 20 fracciones suman 1 | Aprobada |
| 27 | `test_la_matriz_tiene_una_columna_por_hora_de_servicio` | La matriz de elementos tiene las horas 4 a 23 y más de 140 estaciones | Aprobada |
| 28 | `test_la_inercia_baja_al_aumentar_k` | La inercia decrece con k y la silueta está entre -1 y 1 | Aprobada |
| 29 | `test_kmedias_asigna_todas_las_estaciones` | Cada estación recibe un grupo; hay k centroides de 20 dimensiones | Aprobada |
| 30 | `test_kmedias_es_reproducible` | Con la misma semilla, k-medias da los mismos grupos | Aprobada |
| 31 | `test_los_grupos_tienen_sentido` | Portal Américas y Calle 76 caen en grupos distintos, y el del portal es "regreso de la tarde" | Aprobada |
| 32 | `test_jerarquico_y_kmedias_coinciden_en_casi_todo` | Los dos métodos coinciden en más del 90 % de las estaciones | Aprobada |
| 33 | `test_nombrar_grupo` | Un perfil con pico a las 7 se nombra "destino de la mañana" y uno con pico a las 18, "regreso de la tarde" | Aprobada |

## 4. Pruebas de los modelos completos

Además de las automáticas, se ejecutaron los dos scripts completos y se revisaron sus
resultados.

**Actividad 3** (`uv run python -m supervisado.arbol_decision`)

| Medida | Valor |
|---|---|
| Ejemplos de entrenamiento / validación / prueba | 41.720 / 20.860 / 29.800 |
| Profundidad elegida con validación | 20 |
| **Exactitud en los días de prueba (22-31 oct)** | **91,8 %** |
| Exactitud de la línea base | 29,0 % |
| Precisión / exhaustividad de la clase "alta" | 94,4 % / 94,0 % |
| Clase más difícil | "media" (F1 = 0,877): queda entre las otras dos |

**Actividad 4** (`uv run python -m no_supervisado.agrupamiento`)

| Medida | Valor |
|---|---|
| Estaciones agrupadas | 149 |
| k elegido (silueta más alta) | 2 (silueta = 0,551) |
| Grupos | Destino de la mañana: 83 estaciones · Regreso de la tarde: 66 estaciones |
| Coincidencia con el jerárquico (índice de Rand ajustado) | 0,947: solo 2 estaciones cambian de grupo |

## 5. Problemas encontrados durante las pruebas

| Problema | Causa | Solución |
|---|---|---|
| scikit-learn 1.9.1 no cargaba en Windows | El Control inteligente de aplicaciones de Windows 11 bloqueó un archivo DLL de la librería | Se fijó scikit-learn 1.8, que funciona con la protección activa y da resultados idénticos |
| El primer agrupamiento formaba un grupo con una sola "estación" | El *Bicicletero Mirador del Paraíso* tiene 2 salidas en el mes y su perfil es ruido | Se descartan los puntos con menos de 100 salidas en un día hábil |
| Los portales Norte, 80 y Usme no se marcaban como portal | En el archivo se llaman "Cabecera" | `es_portal` reconoce "Portal" y "Cabecera" (prueba 12) |
| Primera versión del árbol con 68 % de exactitud | Sin el tamaño de la estación, el árbol no distinguía estaciones grandes y pequeñas de la misma zona | Se agregó `salidas_diarias_estacion`, calculado solo con los días de entrenamiento |
| La primera versión elegía la profundidad con los datos de prueba | Error de método: la prueba debe usarse una sola vez | Se separó un conjunto de validación (15-21 oct) para elegir la profundidad |
