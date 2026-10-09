"""Actividad 3: árbol de decisión que predice si la demanda de una estación
en una hora será baja, media o alta.

Ejecutar desde la raíz:  uv run python -m supervisado.arbol_decision
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from datos_transmilenio import HORAS_SERVICIO, cargar_datos

CARPETA_RESULTADOS = Path(__file__).parent.parent / "resultados" / "supervisado"

# 1-14 oct entrenamiento, 15-21 validación, 22-31 prueba
ULTIMO_DIA_ENTRENAMIENTO = 14
ULTIMO_DIA_VALIDACION = 21

NIVELES = ["baja", "media", "alta"]
TOLERANCIA = 0.01
SEMILLA = 42


def entropia(etiquetas):
    proporciones = pd.Series(etiquetas).value_counts(normalize=True)
    return float(-(proporciones * np.log2(proporciones)).sum())


def ganancia_de_informacion(etiquetas, condicion):
    etiquetas = pd.Series(etiquetas).reset_index(drop=True)
    condicion = pd.Series(condicion).reset_index(drop=True)
    total = len(etiquetas)
    entropia_hijos = 0.0
    for valor in (True, False):
        grupo = etiquetas[condicion == valor]
        if len(grupo) > 0:
            entropia_hijos += len(grupo) / total * entropia(grupo)
    return entropia(etiquetas) - entropia_hijos


def calcular_umbrales(salidas):
    # terciles: tres grupos del mismo tamaño
    return salidas.quantile([1 / 3, 2 / 3]).to_numpy()


def etiquetar(salidas, umbrales):
    return pd.cut(
        salidas,
        bins=[-np.inf, umbrales[0], umbrales[1], np.inf],
        labels=NIVELES,
    ).astype(str)


def construir_atributos(datos, tamano_estaciones):
    atributos = datos[["hora", "es_portal", "tipo_dia", "zona"]].copy()
    atributos["es_portal"] = atributos["es_portal"].astype(int)
    atributos["salidas_diarias_estacion"] = datos["estacion"].map(tamano_estaciones)
    return pd.get_dummies(atributos, columns=["tipo_dia", "zona"], dtype=int)


def preparar(datos):
    datos = datos[datos["hora"].isin(HORAS_SERVICIO)].reset_index(drop=True)
    conjunto = pd.Series("prueba", index=datos.index)
    conjunto[datos["dia"] <= ULTIMO_DIA_VALIDACION] = "validacion"
    conjunto[datos["dia"] <= ULTIMO_DIA_ENTRENAMIENTO] = "entrenamiento"
    entrenamiento = datos[conjunto == "entrenamiento"]

    # umbrales y tamaño de estación solo con entrenamiento, para no mirar los días de prueba
    umbrales = calcular_umbrales(entrenamiento["salidas"])
    tamano = entrenamiento.groupby("estacion")["salidas"].sum() / ULTIMO_DIA_ENTRENAMIENTO
    etiquetas = etiquetar(datos["salidas"], umbrales)
    atributos = construir_atributos(datos, tamano.round())

    partes = {}
    for nombre in ("entrenamiento", "validacion", "prueba"):
        filtro = conjunto == nombre
        partes[nombre] = (atributos[filtro], etiquetas[filtro])
    return partes, umbrales


def entrenar_arbol(x, y, profundidad):
    arbol = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=profundidad,
        random_state=SEMILLA,
    )
    return arbol.fit(x, y)


def curva_de_profundidad(x_ent, y_ent, x_val, y_val, profundidades):
    filas = []
    for profundidad in profundidades:
        arbol = entrenar_arbol(x_ent, y_ent, profundidad)
        filas.append({
            "profundidad": profundidad,
            "exactitud_entrenamiento": accuracy_score(y_ent, arbol.predict(x_ent)),
            "exactitud_validacion": accuracy_score(y_val, arbol.predict(x_val)),
            "hojas": arbol.get_n_leaves(),
        })
    return pd.DataFrame(filas)


def elegir_profundidad(curva, tolerancia=TOLERANCIA):
    # el árbol más pequeño que quede a menos de `tolerancia` del mejor
    mejor = curva["exactitud_validacion"].max()
    aceptables = curva[curva["exactitud_validacion"] >= mejor - tolerancia]
    return int(aceptables["profundidad"].min())


def exactitud_linea_base(y_ent, y_pru):
    # responder siempre la clase más común
    clase_mas_comun = y_ent.value_counts().idxmax()
    return float((y_pru == clase_mas_comun).mean())


def main():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    partes, umbrales = preparar(cargar_datos())
    x_ent, y_ent = partes["entrenamiento"]
    x_val, y_val = partes["validacion"]
    x_pru, y_pru = partes["prueba"]

    print("=== Datos ===")
    print(f"Entrenamiento (1-{ULTIMO_DIA_ENTRENAMIENTO} oct): {len(x_ent):,} ejemplos")
    print(f"Validación ({ULTIMO_DIA_ENTRENAMIENTO + 1}-{ULTIMO_DIA_VALIDACION} oct): "
          f"{len(x_val):,} ejemplos")
    print(f"Prueba ({ULTIMO_DIA_VALIDACION + 1}-31 oct): {len(x_pru):,} ejemplos")
    print(f"Umbrales: baja <= {umbrales[0]:.0f} < media <= {umbrales[1]:.0f} < alta "
          "(salidas por estación y hora)")

    print("\n=== Entropía y ganancia de información (a mano) ===")
    print(f"Entropía de las etiquetas de entrenamiento: {entropia(y_ent):.3f} bits "
          f"(máximo posible con 3 clases: {np.log2(3):.3f})")
    mediana_tamano = x_ent["salidas_diarias_estacion"].median()
    preguntas = {
        "¿es portal?": x_ent["es_portal"] == 1,
        "¿es domingo o festivo?": x_ent["tipo_dia_domingo_festivo"] == 1,
        "¿hora entre 6 y 20?": x_ent["hora"].between(6, 20),
        "¿estación grande?": x_ent["salidas_diarias_estacion"] > mediana_tamano,
    }
    for pregunta, condicion in preguntas.items():
        ganancia = ganancia_de_informacion(y_ent, condicion)
        print(f"  Ganancia de {pregunta:<24} {ganancia:.3f} bits")

    print("\n=== Elección de la profundidad con los días de validación ===")
    curva = curva_de_profundidad(x_ent, y_ent, x_val, y_val, range(1, 31))
    print(curva.round(3).to_string(index=False))
    profundidad = elegir_profundidad(curva)
    print(f"Profundidad elegida: {profundidad} (la más simple a menos de "
          f"{TOLERANCIA:.0%} de la mejor en validación)")

    arbol = entrenar_arbol(x_ent, y_ent, profundidad)
    predicciones = arbol.predict(x_pru)
    exactitud = accuracy_score(y_pru, predicciones)
    base = exactitud_linea_base(y_ent, y_pru)

    print("\n=== Evaluación final con los días de prueba ===")
    print(f"Exactitud del árbol: {exactitud:.3f}")
    print(f"Exactitud de la línea base (siempre la clase más común): {base:.3f}")
    print(classification_report(y_pru, predicciones, labels=NIVELES, digits=3))

    importancias = pd.Series(arbol.feature_importances_, index=x_ent.columns)
    print("Atributos más importantes:")
    print(importancias.sort_values(ascending=False).head(6).round(3).to_string())

    # árbol corto para poder leer las reglas completas
    arbol_corto = entrenar_arbol(x_ent, y_ent, 3)
    reglas = export_text(arbol_corto, feature_names=list(x_ent.columns))
    exactitud_corto = accuracy_score(y_pru, arbol_corto.predict(x_pru))
    print(f"\n=== Reglas del árbol de profundidad 3 (exactitud en prueba "
          f"{exactitud_corto:.3f}) ===")
    print(reglas)

    guardar_resultados(curva, arbol_corto, x_ent, y_pru, predicciones, reglas,
                       exactitud, base, profundidad, umbrales)
    print(f"Figuras y reglas guardadas en {CARPETA_RESULTADOS}")


def guardar_resultados(curva, arbol_corto, x_ent, y_pru, predicciones, reglas,
                       exactitud, base, profundidad, umbrales):
    figura, eje = plt.subplots(figsize=(8, 5))
    eje.plot(curva["profundidad"], curva["exactitud_entrenamiento"], "o-",
             label=f"Entrenamiento (1-{ULTIMO_DIA_ENTRENAMIENTO} oct)")
    eje.plot(curva["profundidad"], curva["exactitud_validacion"], "s-",
             label=f"Validación ({ULTIMO_DIA_ENTRENAMIENTO + 1}-{ULTIMO_DIA_VALIDACION} oct)")
    eje.axvline(profundidad, color="gray", linestyle="--",
                label=f"Profundidad elegida: {profundidad}")
    eje.set_xlabel("Profundidad máxima del árbol")
    eje.set_ylabel("Exactitud")
    eje.set_title("Exactitud según la profundidad del árbol")
    eje.legend()
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "curva_profundidad.png", dpi=150)
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(22, 9))
    plot_tree(arbol_corto, feature_names=list(x_ent.columns),
              class_names=list(arbol_corto.classes_), filled=True, rounded=True,
              impurity=True, proportion=True, fontsize=10, ax=eje)
    orden = ", ".join(arbol_corto.classes_)
    eje.set_title("Árbol de decisión de profundidad 3 (criterio: entropía)\n"
                  f"value = proporción de cada clase en el nodo, en orden [{orden}]",
                  fontsize=16)
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "arbol_profundidad_3.png", dpi=120)
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_pru, predicciones, labels=NIVELES,
                                            cmap="Blues", ax=eje)
    eje.set_title(f"Matriz de confusión, días de prueba (exactitud {exactitud:.1%})")
    eje.set_xlabel("Predicción del árbol")
    eje.set_ylabel("Valor real")
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "matriz_confusion.png", dpi=150)
    plt.close(figura)

    (CARPETA_RESULTADOS / "reglas_profundidad_3.txt").write_text(reglas, encoding="utf-8")
    curva.round(4).to_csv(CARPETA_RESULTADOS / "curva_profundidad.csv", index=False)
    resumen = (
        f"Umbrales: baja <= {umbrales[0]:.0f} < media <= {umbrales[1]:.0f} < alta\n"
        f"Profundidad elegida: {profundidad}\n"
        f"Exactitud en prueba: {exactitud:.4f}\n"
        f"Exactitud de la línea base: {base:.4f}\n"
    )
    (CARPETA_RESULTADOS / "metricas.txt").write_text(resumen, encoding="utf-8")


if __name__ == "__main__":
    main()
