import math

import pandas as pd
import pytest

from datos_transmilenio import cargar_datos
from supervisado.arbol_decision import (
    NIVELES,
    calcular_umbrales,
    elegir_profundidad,
    entrenar_arbol,
    entropia,
    etiquetar,
    exactitud_linea_base,
    ganancia_de_informacion,
    preparar,
)


@pytest.fixture(scope="module")
def partes():
    partes, _ = preparar(cargar_datos())
    return partes


# casos que se pueden calcular a mano

def test_entropia_de_un_grupo_puro_es_cero():
    assert entropia(["alta", "alta", "alta"]) == 0


def test_entropia_de_dos_clases_mitad_y_mitad_es_un_bit():
    assert entropia(["alta", "baja", "alta", "baja"]) == pytest.approx(1.0)


def test_entropia_maxima_con_tres_clases():
    assert entropia(["baja", "media", "alta"]) == pytest.approx(math.log2(3))


def test_ganancia_de_una_pregunta_que_separa_perfecto():
    etiquetas = ["alta", "alta", "baja", "baja"]
    condicion = [True, True, False, False]
    assert ganancia_de_informacion(etiquetas, condicion) == pytest.approx(1.0)


def test_ganancia_de_una_pregunta_inutil_es_cero():
    etiquetas = ["alta", "baja", "alta", "baja"]
    condicion = [True, True, False, False]
    assert ganancia_de_informacion(etiquetas, condicion) == pytest.approx(0.0)



def test_umbrales_y_etiquetas():
    salidas = pd.Series(range(1, 301))
    umbrales = calcular_umbrales(salidas)
    etiquetas = etiquetar(salidas, umbrales)
    assert list(etiquetas.value_counts().sort_index().index) == sorted(NIVELES)
    assert etiquetas.iloc[0] == "baja" and etiquetas.iloc[-1] == "alta"


def test_los_conjuntos_no_se_mezclan(partes):
    tamanos = [len(x) for x, _ in partes.values()]
    indices = [set(x.index) for x, _ in partes.values()]
    assert all(t > 0 for t in tamanos)
    assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])


def test_el_entrenamiento_tiene_las_tres_clases_balanceadas(partes):
    _, y_ent = partes["entrenamiento"]
    proporciones = y_ent.value_counts(normalize=True)
    assert set(proporciones.index) == set(NIVELES)
    assert proporciones.between(0.30, 0.37).all()



def test_elegir_profundidad_prefiere_el_arbol_mas_simple():
    curva = pd.DataFrame({
        "profundidad": [1, 2, 3, 4],
        "exactitud_validacion": [0.50, 0.80, 0.805, 0.81],
    })
    assert elegir_profundidad(curva, tolerancia=0.01) == 2


def test_el_arbol_supera_la_linea_base(partes):
    x_ent, y_ent = partes["entrenamiento"]
    x_pru, y_pru = partes["prueba"]
    arbol = entrenar_arbol(x_ent, y_ent, profundidad=8)
    exactitud = (arbol.predict(x_pru) == y_pru.to_numpy()).mean()
    assert exactitud > exactitud_linea_base(y_ent, y_pru) + 0.30


def test_el_arbol_es_reproducible(partes):
    x_ent, y_ent = partes["entrenamiento"]
    x_val, _ = partes["validacion"]
    a = entrenar_arbol(x_ent, y_ent, profundidad=10).predict(x_val)
    b = entrenar_arbol(x_ent, y_ent, profundidad=10).predict(x_val)
    assert (a == b).all()
