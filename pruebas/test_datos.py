"""Pruebas de la carga y limpieza de datos (datos_transmilenio.py)."""

import pytest

from datos_transmilenio import (
    MINIMO_SALIDAS_DIARIAS,
    a_formato_largo,
    cargar_crudo,
    cargar_datos,
    quitar_codigo,
    tipo_de_dia,
)


@pytest.fixture(scope="module")
def crudo():
    return cargar_crudo()


@pytest.fixture(scope="module")
def datos():
    return cargar_datos()


def test_el_crudo_no_trae_filas_vacias(crudo):
    assert crudo["Estación"].notna().all()
    assert not any(c.startswith("Unnamed") for c in crudo.columns)


def test_el_crudo_trae_los_31_dias_de_octubre(crudo):
    columnas_dia = [c for c in crudo.columns if c.startswith("DÍA")]
    assert len(columnas_dia) == 31


def test_quitar_codigo():
    assert quitar_codigo("(06000) Portal Eldorado") == "Portal Eldorado"
    assert quitar_codigo("(34) Zona H Caracas Sur") == "Zona H Caracas Sur"
    assert quitar_codigo("Calle 57") == "Calle 57"


@pytest.mark.parametrize("dia, esperado", [
    (1, "habil"),             # martes 1 de octubre de 2024
    (5, "sabado"),
    (6, "domingo_festivo"),   # domingo
    (14, "domingo_festivo"),  # lunes festivo (Día de la Raza trasladado)
    (15, "habil"),
])
def test_tipo_de_dia(dia, esperado):
    assert tipo_de_dia(dia) == esperado


def test_el_formato_largo_no_pierde_salidas():
    """Pasar de ancho a largo no puede crear ni perder pasajeros."""
    crudo = cargar_crudo()
    columnas_dia = [c for c in crudo.columns if c.startswith("DÍA")]
    total_crudo = crudo[columnas_dia].fillna(0).round().sum().sum()
    largo = a_formato_largo(crudo)
    # La única diferencia permitida es la de los puntos descartados por casi no tener salidas.
    assert total_crudo - largo["salidas"].sum() < 1_000
    assert largo["salidas"].sum() > 58_000_000


def test_no_hay_valores_faltantes_ni_negativos(datos):
    assert datos.notna().all().all()
    assert (datos["salidas"] >= 0).all()


def test_se_descartan_los_puntos_que_no_son_estaciones(datos):
    habiles = datos[datos["tipo_dia"] == "habil"]
    promedio = habiles.groupby(["estacion", "dia"])["salidas"].sum().groupby("estacion").mean()
    assert (promedio >= MINIMO_SALIDAS_DIARIAS).all()
    assert "Bicicletero Mirador del Paraíso" not in set(datos["estacion"])


def test_los_portales_se_marcan_incluidas_las_cabeceras(datos):
    portales = set(datos.loc[datos["es_portal"], "estacion"])
    assert "Portal Américas" in portales
    assert "Cabecera Autopista Norte" in portales
    assert "Calle 57" not in portales


def test_un_dia_habil_mueve_mas_gente_que_un_domingo(datos):
    """Prueba de cordura con un hecho conocido del sistema."""
    por_dia = datos.groupby(["dia", "tipo_dia"])["salidas"].sum().reset_index()
    promedio = por_dia.groupby("tipo_dia")["salidas"].mean()
    assert promedio["habil"] > promedio["sabado"] > promedio["domingo_festivo"]
