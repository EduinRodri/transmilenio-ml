"""Actividad 4: agrupamiento de estaciones según la hora en que la gente se baja.

Ejecutar desde la raíz:  uv run python -m no_supervisado.agrupamiento
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from datos_transmilenio import HORAS_SERVICIO, cargar_datos

CARPETA_RESULTADOS = Path(__file__).parent.parent / "resultados" / "no_supervisado"
VALORES_DE_K = range(2, 9)
SEMILLA = 42


def perfiles_horarios(datos):
    # fracción de las salidas de un día hábil en cada hora; cada fila suma 1,
    # así se compara la forma del día y no el tamaño de la estación
    habiles = datos[(datos["tipo_dia"] == "habil") & datos["hora"].isin(HORAS_SERVICIO)]
    promedio = habiles.groupby(["estacion", "hora"])["salidas"].mean().unstack(fill_value=0)
    return promedio.div(promedio.sum(axis=1), axis=0)


def distancia_euclidiana(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum()))


def evaluar_k(perfiles, valores_de_k=VALORES_DE_K):
    filas = []
    for k in valores_de_k:
        modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(perfiles)
        filas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silueta": silhouette_score(perfiles, modelo.labels_),
        })
    return pd.DataFrame(filas)


def agrupar_kmedias(perfiles, k):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(perfiles)
    return modelo.labels_, modelo.cluster_centers_


def nombrar_grupo(centroide, horas):
    hora_pico = horas[int(np.argmax(centroide))]
    if hora_pico < 10:
        return f"Destino de la mañana (pico {hora_pico}:00)"
    if hora_pico >= 15:
        return f"Regreso de la tarde (pico {hora_pico}:00)"
    return f"Mixto (pico {hora_pico}:00)"


def agrupar_jerarquico(perfiles, k):
    enlaces = linkage(perfiles, method="ward")
    etiquetas = fcluster(enlaces, t=k, criterion="maxclust") - 1  # fcluster numera desde 1
    return etiquetas, enlaces


def main():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    datos = cargar_datos()
    perfiles = perfiles_horarios(datos)
    horas = list(perfiles.columns)

    print("=== Matriz de elementos ===")
    print(f"{perfiles.shape[0]} estaciones x {perfiles.shape[1]} horas (de {horas[0]}:00 "
          f"a {horas[-1]}:00), días hábiles de octubre de 2024")

    print("\n=== Distancia euclidiana entre perfiles (ejemplo) ===")
    for a, b in [("Portal Américas", "Portal Suba"), ("Portal Américas", "Calle 76 - San Felipe")]:
        print(f"  d({a}, {b}) = {distancia_euclidiana(perfiles.loc[a], perfiles.loc[b]):.3f}")

    print("\n=== Elección de k (método del codo y silueta) ===")
    evaluacion = evaluar_k(perfiles)
    print(evaluacion.round(3).to_string(index=False))
    k = int(evaluacion.loc[evaluacion["silueta"].idxmax(), "k"])
    print(f"k elegido: {k} (silueta más alta)")

    etiquetas_km, centroides = agrupar_kmedias(perfiles, k)
    nombres = [nombrar_grupo(c, horas) for c in centroides]

    print("\n=== Grupos de k-medias ===")
    salidas_diarias = (
        datos[datos["tipo_dia"] == "habil"].groupby(["estacion", "dia"])["salidas"].sum()
        .groupby("estacion").mean()
    )
    tabla = pd.DataFrame({
        "estacion": perfiles.index,
        "grupo": [nombres[e] for e in etiquetas_km],
        "salidas_dia_habil": salidas_diarias.reindex(perfiles.index).round().astype(int).values,
    })
    zona = datos.drop_duplicates("estacion").set_index("estacion")["zona"]
    tabla["zona"] = tabla["estacion"].map(zona)
    for nombre in nombres:
        miembros = tabla[tabla["grupo"] == nombre].sort_values("salidas_dia_habil", ascending=False)
        print(f"\n{nombre}: {len(miembros)} estaciones")
        print("  Las más grandes:", ", ".join(miembros["estacion"].head(6)))
        print("  Zonas más frecuentes:",
              miembros["zona"].value_counts().head(3).to_dict())

    etiquetas_jer, enlaces = agrupar_jerarquico(perfiles, k)
    acuerdo = adjusted_rand_score(etiquetas_km, etiquetas_jer)
    print("\n=== Comparación k-medias vs. jerárquico (Ward) ===")
    print(pd.crosstab(pd.Series([nombres[e] for e in etiquetas_km], name="k-medias"),
                      pd.Series(etiquetas_jer, name="jerárquico")))
    print(f"Índice de Rand ajustado: {acuerdo:.3f} (1 = los dos métodos coinciden del todo)")

    guardar_resultados(perfiles, evaluacion, k, etiquetas_km, centroides, nombres,
                       enlaces, tabla, acuerdo)
    print(f"\nFiguras y tabla guardadas en {CARPETA_RESULTADOS}")


def guardar_resultados(perfiles, evaluacion, k, etiquetas_km, centroides, nombres,
                       enlaces, tabla, acuerdo):
    horas = list(perfiles.columns)

    figura, (eje1, eje2) = plt.subplots(1, 2, figsize=(11, 4.5))
    eje1.plot(evaluacion["k"], evaluacion["inercia"], "o-")
    eje1.set_title("Método del codo")
    eje1.set_xlabel("Número de grupos (k)")
    eje1.set_ylabel("Inercia")
    eje2.plot(evaluacion["k"], evaluacion["silueta"], "s-", color="tab:orange")
    eje2.axvline(k, color="gray", linestyle="--", label=f"k elegido: {k}")
    eje2.set_title("Coeficiente de silueta")
    eje2.set_xlabel("Número de grupos (k)")
    eje2.set_ylabel("Silueta promedio")
    eje2.legend()
    for eje in (eje1, eje2):
        eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "codo_y_silueta.png", dpi=150)
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(10, 5.5))
    colores = plt.cm.tab10.colors
    for grupo in range(k):
        for _, fila in perfiles[etiquetas_km == grupo].iterrows():
            eje.plot(horas, fila.values, color=colores[grupo], alpha=0.12, linewidth=0.8)
        eje.plot(horas, centroides[grupo], color=colores[grupo], linewidth=3,
                 label=f"{nombres[grupo]} — {(etiquetas_km == grupo).sum()} estaciones")
    eje.set_xticks(horas)
    eje.set_xlabel("Hora del día")
    eje.set_ylabel("Fracción de las salidas del día")
    eje.set_title("Perfil horario de cada grupo (línea gruesa = centroide de k-medias)")
    eje.legend()
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "perfiles_por_grupo.png", dpi=150)
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(12, 5.5))
    dendrogram(enlaces, truncate_mode="lastp", p=30, ax=eje, color_threshold=None)
    eje.set_title(f"Dendrograma (Ward), últimas 30 uniones · coincide con k-medias: "
                  f"índice de Rand ajustado {acuerdo:.2f}")
    eje.set_xlabel("Estaciones o grupos de estaciones (entre paréntesis, cuántas)")
    eje.set_ylabel("Distancia de unión")
    figura.tight_layout()
    figura.savefig(CARPETA_RESULTADOS / "dendrograma.png", dpi=150)
    plt.close(figura)

    tabla.sort_values(["grupo", "salidas_dia_habil"], ascending=[True, False]).to_csv(
        CARPETA_RESULTADOS / "grupos_de_estaciones.csv", index=False, encoding="utf-8-sig")
    evaluacion.round(4).to_csv(CARPETA_RESULTADOS / "eleccion_de_k.csv", index=False)


if __name__ == "__main__":
    main()
