# Aprendizaje automático con datos reales de TransMilenio

Actividades 3 y 4 de Inteligencia Artificial — Corporación Universitaria Iberoamericana, 2026-2.

**Integrantes:** Eduin David Rodríguez Vidal · Erika Milena Bernal · Juan Diego Quintero Zambrano

El proyecto continúa el sistema de transporte masivo de las actividades anteriores (rutas en
TransMilenio). Esta vez se aprenden patrones a partir de los **registros reales de salidas del
sistema troncal en octubre de 2024**, publicados por TransMilenio S.A. en Datos Abiertos Bogotá.

| | Actividad 3 — Supervisado | Actividad 4 — No supervisado |
|---|---|---|
| Pregunta | Dada una estación, una hora y un tipo de día, ¿la demanda será baja, media o alta? | ¿Qué tipos de estaciones hay según la hora en que la gente se baja en ellas? |
| Técnica | Árbol de decisión con entropía (Palma, cap. 17) | k-medias y agrupamiento jerárquico de Ward (Palma, cap. 16) |
| Código | [`supervisado/arbol_decision.py`](supervisado/arbol_decision.py) | [`no_supervisado/agrupamiento.py`](no_supervisado/agrupamiento.py) |
| Resultado | **91,8 %** de exactitud en días que el modelo nunca vio (línea base: 29,0 %) | **2 grupos**: estaciones de destino de la mañana (83) y de regreso de la tarde (66) |

## Cómo ejecutarlo

Requiere [uv](https://docs.astral.sh/uv/) (instala Python 3.13 y las librerías solo).

```bash
uv sync                                        # instala dependencias
uv run python datos_transmilenio.py            # resumen de los datos
uv run python -m supervisado.arbol_decision    # Actividad 3
uv run python -m no_supervisado.agrupamiento   # Actividad 4
uv run pytest -v                               # 33 pruebas
```

Cada script imprime sus resultados en consola y guarda las figuras en `resultados/`.

## Estructura

```
datos/
  salidas_troncal_2024_10.csv   Datos originales de TransMilenio, sin modificar
datos_transmilenio.py           Carga, limpieza y paso a formato largo (lo usan los dos modelos)
supervisado/arbol_decision.py   Actividad 3
no_supervisado/agrupamiento.py  Actividad 4
pruebas/                        Pruebas automáticas con pytest
resultados/                     Figuras, reglas y tablas que generan los scripts
docs/
  descripcion-datos.md          Fuente, columnas, limpieza y limitaciones de los datos
  pruebas-realizadas.md         Qué se probó y con qué resultado
  mapa-conceptual-actividad-3   Mapa conceptual del aprendizaje supervisado (PNG y PDF)
```

## Resultados principales

**Actividad 3.** El árbol aprende con los días 1 al 14 de octubre, elige su profundidad con los
días 15 al 21 y se evalúa una sola vez con los días 22 al 31. Los atributos que más pesan son el
tamaño de la estación y la hora. Ver `resultados/supervisado/`: árbol de profundidad 3 con sus
reglas, curva de exactitud según la profundidad y matriz de confusión.

**Actividad 4.** Cada estación se describe por la forma de su día hábil (qué fracción de sus
salidas ocurre en cada hora). k-medias separa las estaciones donde la gente llega por la mañana
(Calle 57, Calle 76, Calle 100, Av. Jiménez: zonas de trabajo y estudio) de aquellas donde llega
por la tarde (todos los portales y las estaciones del sur: zonas de vivienda). El agrupamiento
jerárquico llega casi al mismo resultado (índice de Rand ajustado 0,95). Ver
`resultados/no_supervisado/`.

## Fuentes

- TransMilenio S.A. (2024). *Consolidado de salidas sistema troncal por franja horaria — Octubre
  2024* [Conjunto de datos]. Datos Abiertos Bogotá. Licencia CC BY 4.0.
  https://datosabiertos.bogota.gov.co/dataset/consolidado-de-salidas-sistema-troncal-por-franja-horaria
- Vila Miranda, A. y Delgado Calvo-Flores, M. (2008). Técnicas de agrupamiento. En J. T. Palma
  Méndez y R. Marín Morales (Eds.), *Inteligencia artificial: métodos, técnicas y aplicaciones*
  (pp. 691-724). McGraw-Hill España.
- Ferri Ramírez, C. y Ramírez Quintana, M. J. (2008). Aprendizaje de árboles y reglas de
  decisión. En J. T. Palma Méndez y R. Marín Morales (Eds.), *Inteligencia artificial: métodos,
  técnicas y aplicaciones* (pp. 725-762). McGraw-Hill España.
