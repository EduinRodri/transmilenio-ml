"""Carga y limpieza de las salidas del sistema troncal de TransMilenio (octubre 2024)."""

from datetime import date
from pathlib import Path

import pandas as pd

RUTA_CSV = Path(__file__).parent / "datos" / "salidas_troncal_2024_10.csv"
ANIO = 2024
MES = 10

# el 12 de octubre (Día de la Raza) se pasó al lunes 14
FESTIVOS = {date(2024, 10, 14)}

HORAS_SERVICIO = range(4, 24)

# el bicicletero y el corral del Portal Eldorado no son estaciones de pasajeros
MINIMO_SALIDAS_DIARIAS = 100


def cargar_crudo(ruta=RUTA_CSV):
    datos = pd.read_csv(ruta, sep=";", encoding="cp1252", low_memory=False)
    # el archivo trae miles de filas vacías al final y dos columnas sin nombre
    datos = datos.dropna(subset=["Estación"])
    datos = datos.loc[:, ~datos.columns.str.startswith("Unnamed")]
    return datos


def quitar_codigo(texto):
    # "(06000) Portal Eldorado" -> "Portal Eldorado"
    return texto.split(") ", 1)[1] if texto.startswith("(") else texto


def tipo_de_dia(dia):
    fecha = date(ANIO, MES, dia)
    if fecha in FESTIVOS or fecha.weekday() == 6:
        return "domingo_festivo"
    if fecha.weekday() == 5:
        return "sabado"
    return "habil"


def a_formato_largo(crudo):
    """Una fila por estación, día y hora (suma accesos y franjas de 15 min)."""
    columnas_dia = [c for c in crudo.columns if c.startswith("DÍA")]
    largo = crudo.melt(
        id_vars=["Línea", "Estación", "INTERVALO"],
        value_vars=columnas_dia,
        var_name="columna_dia",
        value_name="salidas",
    )
    largo["dia"] = largo["columna_dia"].str.extract(r"(\d+)").astype(int)
    largo["hora"] = largo["INTERVALO"].str[:2].astype(int)
    largo["zona"] = largo["Línea"].map(quitar_codigo)
    largo["estacion"] = largo["Estación"].map(quitar_codigo)
    # hay conteos con decimales
    largo["salidas"] = largo["salidas"].fillna(0).round().astype(int)

    por_hora = (
        largo.groupby(["zona", "estacion", "dia", "hora"], as_index=False)["salidas"].sum()
    )
    por_hora["tipo_dia"] = por_hora["dia"].map(tipo_de_dia)
    # los portales Norte, 80 y Usme aparecen como "Cabecera"
    por_hora["es_portal"] = por_hora["estacion"].str.startswith(("Portal", "Cabecera"))
    return quitar_estaciones_minimas(por_hora)


def quitar_estaciones_minimas(por_hora):
    habiles = por_hora[por_hora["tipo_dia"] == "habil"]
    promedio_diario = habiles.groupby(["estacion", "dia"])["salidas"].sum().groupby("estacion").mean()
    validas = promedio_diario[promedio_diario >= MINIMO_SALIDAS_DIARIAS].index
    return por_hora[por_hora["estacion"].isin(validas)].reset_index(drop=True)


def cargar_datos(ruta=RUTA_CSV):
    return a_formato_largo(cargar_crudo(ruta))


if __name__ == "__main__":
    datos = cargar_datos()
    print(f"Filas: {len(datos):,}")
    print(f"Estaciones: {datos['estacion'].nunique()}  Zonas: {datos['zona'].nunique()}")
    print(f"Salidas en el mes: {datos['salidas'].sum():,}")
    print("\nSalidas promedio por día según el tipo de día:")
    por_dia = datos.groupby(["dia", "tipo_dia"])["salidas"].sum().reset_index()
    print(por_dia.groupby("tipo_dia")["salidas"].mean().round().astype(int))
