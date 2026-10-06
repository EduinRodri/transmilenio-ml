# Descripción de los datos

## Fuente

| | |
|---|---|
| Conjunto de datos | Consolidado de salidas sistema troncal por franja horaria — Octubre 2024 |
| Publicado por | Empresa de Transporte del Tercer Milenio — TransMilenio S.A. |
| Portal | Datos Abiertos Bogotá — https://datosabiertos.bogota.gov.co/dataset/consolidado-de-salidas-sistema-troncal-por-franja-horaria |
| Licencia | Creative Commons Atribución 4.0 (CC BY 4.0): se puede usar citando la fuente |
| Archivo | `datos/salidas_troncal_2024_10.csv` (11,1 MB), copia sin modificar del original |
| Periodo | 1 al 31 de octubre de 2024 |

**Por qué octubre de 2024.** Es el mes completo más reciente publicado y es un mes "normal": no
tiene vacaciones escolares largas ni temporada navideña, e incluye un festivo (lunes 14), que
sirve para comprobar que el modelo distingue los días de poca demanda.

**Relación con el proyecto.** En las actividades anteriores el equipo construyó un sistema que
calcula rutas en TransMilenio. Estos datos agregan lo que ese sistema no sabía: cuánta gente usa
cada estación y a qué hora.

## Estructura del archivo original

Formato CSV separado por punto y coma (`;`) y codificado en Windows-1252.

| Columna | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `Línea` | texto | Zona troncal de la estación, con su código | `(11) Zona K Calle 26` |
| `Estación` | texto | Estación troncal, con su código | `(06000) Portal Eldorado` |
| `Acceso de Estación` | texto | Puerta o plataforma de la estación por donde sale el usuario | `(01) PLAT2 ALIM-DESAL ...` |
| `MES` | texto | Mes de la medición | `OCTUBRE` |
| `INTERVALO` | texto `HH:MM` | Inicio de la franja de 15 minutos | `07:15` |
| `DÍA 01` … `DÍA 31` | número | Salidas registradas en esa franja ese día | `214` |
| `Total general` | número | Suma de los 31 días | `5180` |

Cada fila es una combinación de estación, acceso y franja de 15 minutos. El archivo tiene 50.322
filas con datos: 151 estaciones, 568 accesos, 13 zonas y 89 franjas distintas.

**Qué es una "salida".** Es el registro de un usuario que sale del sistema por el torniquete de
una estación troncal. Por eso dice dónde y cuándo **termina** un viaje.

## Limpieza y transformación

Todo se hace en [`datos_transmilenio.py`](../datos_transmilenio.py). El archivo original no se
modifica.

1. **Filas vacías.** El archivo trae 46.133 filas vacías al final y dos columnas sin nombre. Se
   eliminan.
2. **Códigos.** Se quita el código entre paréntesis de zonas y estaciones:
   `(06000) Portal Eldorado` pasa a `Portal Eldorado`.
3. **Decimales.** Algunos conteos traen decimales. Como son personas, se redondean.
4. **Formato largo.** Se pasa de una columna por día a una fila por estación, día y hora. Se
   suman los accesos de cada estación y las cuatro franjas de cada hora.
5. **Atributos derivados.**
   - `tipo_dia`: `habil`, `sabado` o `domingo_festivo`. Se calcula con la fecha (el 1 de octubre
     de 2024 fue martes) y la lista de festivos (lunes 14 de octubre, Día de la Raza trasladado).
   - `es_portal`: verdadero para los portales. En el archivo, los portales Norte, 80 y Usme se
     llaman "Cabecera", así que también cuentan.
6. **Puntos que no son estaciones.** Se descartan los que registran menos de 100 salidas en un
   día hábil promedio: *Bicicletero Mirador del Paraíso* (2 salidas en todo el mes) y *Corral
   Portal Dorado* (13 al día). Quedan **149 estaciones**.
7. **Horario de servicio.** Los modelos usan solo de 4:00 a. m. a 11:59 p. m. Lo registrado de
   madrugada es menos del 0,2 % del mes.

Resultado: una tabla de 105.803 filas con estas columnas.

| Columna | Descripción |
|---|---|
| `zona` | Zona troncal (13 valores) |
| `estacion` | Nombre de la estación (149 valores) |
| `dia` | Día de octubre (1 a 31) |
| `hora` | Hora del día (0 a 23) |
| `salidas` | Salidas registradas en esa estación, ese día y esa hora |
| `tipo_dia` | `habil`, `sabado` o `domingo_festivo` |
| `es_portal` | Si la estación es un portal o cabecera |

## Validación contra la realidad

| Tipo de día | Salidas promedio por día |
|---|---|
| Hábil | 2.182.591 |
| Sábado | 1.638.108 |
| Domingo o festivo | 873.027 |

Las cifras son coherentes con el volumen conocido del sistema troncal. El festivo del 14 de
octubre (781.959 salidas) se parece a un domingo y no a un lunes, lo que confirma que la
clasificación de días es correcta.

## Cómo usa los datos cada modelo

**Actividad 3 — Árbol de decisión (supervisado).** Cada ejemplo es una estación en una hora de
un día. La **etiqueta** es el nivel de demanda: `baja`, `media` o `alta`, según los terciles de
las salidas en los días de entrenamiento (baja ≤ 167 < media ≤ 507 < alta). Los **atributos**
son la hora, el tipo de día, la zona, si es portal y el tamaño de la estación (sus salidas
diarias promedio en los días de entrenamiento). Los días se reparten en orden de calendario:
entrenamiento del 1 al 14, validación del 15 al 21 y prueba del 22 al 31.

**Actividad 4 — Agrupamiento (no supervisado).** No hay etiquetas. Cada estación se describe con
un vector de 20 números: la fracción de sus salidas de un día hábil promedio que ocurre en cada
hora, de 4 a 23. Se usan fracciones y no conteos para comparar la *forma* del día y no el
tamaño de la estación.

## Limitaciones

- **Solo salidas.** El archivo no dice dónde empezó cada viaje, así que no permite reconstruir
  orígenes y destinos.
- **Solo el sistema troncal.** No incluye el SITP zonal. Sí incluye la estación de conexión con el TransMiCable (Cable Portal Tunal).
- **Un solo mes.** Los patrones de octubre pueden cambiar en vacaciones o en diciembre.
- **Salidas sin torniquete.** Si una estación tiene salidas sin validación o hay evasión, esas
  personas no quedan registradas.
