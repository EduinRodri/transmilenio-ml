"""Carga y limpieza de los datos de salidas del sistema troncal de TransMilenio.

Fuente: "Consolidado de salidas sistema troncal por franja horaria", TransMilenio S.A.,
Datos Abiertos Bogotá, licencia CC BY 4.0. Archivo de octubre de 2024.

El archivo original viene en formato ancho: una fila por estación, acceso y franja de
15 minutos, y una columna por cada día del mes. Este módulo lo pasa a formato largo
(una fila por estación, día y hora), que es el que usan los dos modelos.
"""

from datetime import date
from pathlib import Path

import pandas as pd

RUTA_CSV = Path(__file__).parent / "datos" / "salidas_troncal_2024_10.csv"
ANIO = 2024
MES = 10

# Festivos de Colombia en octubre de 2024: el Día de la Raza (12 de octubre)
# se trasladó al lunes 14 por la Ley Emiliani.
FESTIVOS = {date(2024, 10, 14)}

# El servicio troncal opera de 4:00 a. m. a 11:00 p. m. Lo que se registra de madrugada
# es casi nada (menos del 0,2 % del mes) y los modelos lo dejan por fuera.
HORAS_SERVICIO = range(4, 24)

# Puntos con menos de 100 salidas en un día hábil promedio no son estaciones de pasajeros
# (en octubre de 2024: un bicicletero y un corral del Portal Eldorado). Se descartan.
MINIMO_SALIDAS_DIARIAS = 100


def cargar_crudo(ruta=RUTA_CSV):
    """Lee el CSV tal como lo publica TransMilenio y quita las filas vacías."""
    datos = pd.read_csv(ruta, sep=";", encoding="cp1252", low_memory=False)
    # El archivo trae miles de filas vacías al final y dos columnas sin nombre.
    datos = datos.dropna(subset=["Estación"])
    datos = datos.loc[:, ~datos.columns.str.startswith("Unnamed")]
    return datos


def quitar_codigo(texto):
    """'(06000) Portal Eldorado' -> 'Portal Eldorado'."""
    return texto.split(") ", 1)[1] if texto.startswith("(") else texto


def tipo_de_dia(dia):
    """Clasifica un día de octubre de 2024 en 'habil', 'sabado' o 'domingo_festivo'."""
    fecha = date(ANIO, MES, dia)
    if fecha in FESTIVOS or fecha.weekday() == 6:
        return "domingo_festivo"
    if fecha.weekday() == 5:
        return "sabado"
    return "habil"


def a_formato_largo(crudo):
    """Pasa de una columna por día a una fila por estación, día y hora.

    Suma los accesos de cada estación y junta las cuatro franjas de 15 minutos
    de cada hora.
    """
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
    # Algunos conteos vienen con decimales; las salidas son personas, así que se redondean.
    largo["salidas"] = largo["salidas"].fillna(0).round().astype(int)

    por_hora = (
        largo.groupby(["zona", "estacion", "dia", "hora"], as_index=False)["salidas"].sum()
    )
    por_hora["tipo_dia"] = por_hora["dia"].map(tipo_de_dia)
    # En el archivo, los portales Norte, 80 y Usme aparecen como "Cabecera".
    por_hora["es_portal"] = por_hora["estacion"].str.startswith(("Portal", "Cabecera"))
    return quitar_estaciones_minimas(por_hora)


def quitar_estaciones_minimas(por_hora):
    """Descarta los puntos que casi no registran salidas (ver MINIMO_SALIDAS_DIARIAS)."""
    habiles = por_hora[por_hora["tipo_dia"] == "habil"]
    promedio_diario = habiles.groupby(["estacion", "dia"])["salidas"].sum().groupby("estacion").mean()
    validas = promedio_diario[promedio_diario >= MINIMO_SALIDAS_DIARIAS].index
    return por_hora[por_hora["estacion"].isin(validas)].reset_index(drop=True)


def cargar_datos(ruta=RUTA_CSV):
    """Atajo: lee el CSV y lo devuelve ya limpio y en formato largo."""
    return a_formato_largo(cargar_crudo(ruta))


if __name__ == "__main__":
    datos = cargar_datos()
    print(f"Filas: {len(datos):,}")
    print(f"Estaciones: {datos['estacion'].nunique()}  Zonas: {datos['zona'].nunique()}")
    print(f"Salidas en el mes: {datos['salidas'].sum():,}")
    print("\nSalidas promedio por día según el tipo de día:")
    por_dia = datos.groupby(["dia", "tipo_dia"])["salidas"].sum().reset_index()
    print(por_dia.groupby("tipo_dia")["salidas"].mean().round().astype(int))
