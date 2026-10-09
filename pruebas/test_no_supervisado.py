import numpy as np
import pytest

from datos_transmilenio import HORAS_SERVICIO, cargar_datos
from no_supervisado.agrupamiento import (
    agrupar_jerarquico,
    agrupar_kmedias,
    distancia_euclidiana,
    evaluar_k,
    nombrar_grupo,
    perfiles_horarios,
)


@pytest.fixture(scope="module")
def perfiles():
    return perfiles_horarios(cargar_datos())


def test_distancia_euclidiana_a_mano():
    assert distancia_euclidiana([0, 0], [3, 4]) == pytest.approx(5.0)
    assert distancia_euclidiana([1, 2, 3], [1, 2, 3]) == 0


def test_cada_perfil_suma_uno(perfiles):
    assert np.allclose(perfiles.sum(axis=1), 1.0)


def test_la_matriz_tiene_una_columna_por_hora_de_servicio(perfiles):
    assert list(perfiles.columns) == list(HORAS_SERVICIO)
    assert perfiles.shape[0] > 140


def test_la_inercia_baja_al_aumentar_k(perfiles):
    evaluacion = evaluar_k(perfiles, range(2, 6))
    assert evaluacion["inercia"].is_monotonic_decreasing
    assert evaluacion["silueta"].between(-1, 1).all()


def test_kmedias_asigna_todas_las_estaciones(perfiles):
    etiquetas, centroides = agrupar_kmedias(perfiles, 2)
    assert len(etiquetas) == len(perfiles)
    assert set(etiquetas) == {0, 1}
    assert centroides.shape == (2, perfiles.shape[1])


def test_kmedias_es_reproducible(perfiles):
    a, _ = agrupar_kmedias(perfiles, 3)
    b, _ = agrupar_kmedias(perfiles, 3)
    assert (a == b).all()


def test_los_grupos_tienen_sentido(perfiles):
    etiquetas, centroides = agrupar_kmedias(perfiles, 2)
    grupo = dict(zip(perfiles.index, etiquetas))
    assert grupo["Portal Américas"] != grupo["Calle 76 - San Felipe"]
    nombre_portal = nombrar_grupo(centroides[grupo["Portal Américas"]], list(perfiles.columns))
    assert nombre_portal.startswith("Regreso de la tarde")


def test_jerarquico_y_kmedias_coinciden_en_casi_todo(perfiles):
    etiquetas_km, _ = agrupar_kmedias(perfiles, 2)
    etiquetas_jer, _ = agrupar_jerarquico(perfiles, 2)
    # Los números de grupo pueden venir invertidos, así que se mira la mejor coincidencia.
    coincidencia = max((etiquetas_km == etiquetas_jer).mean(),
                       (etiquetas_km != etiquetas_jer).mean())
    assert coincidencia > 0.9


def test_nombrar_grupo():
    horas = list(HORAS_SERVICIO)
    pico_manana = np.zeros(len(horas))
    pico_manana[horas.index(7)] = 1
    pico_tarde = np.zeros(len(horas))
    pico_tarde[horas.index(18)] = 1
    assert nombrar_grupo(pico_manana, horas).startswith("Destino de la mañana")
    assert nombrar_grupo(pico_tarde, horas).startswith("Regreso de la tarde")
