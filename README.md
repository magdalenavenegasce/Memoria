# Sistema de Análisis de Marcha y Riesgo de Caídas — OpenCap Offline

Plataforma computacional para el análisis biomecánico de la marcha humana a partir de datos
de captura de movimiento sin marcadores (OpenCap), que integra un score clínico
espaciotemporal y un score independiente de dinámica no lineal (entropía) para apoyar la
estimación objetiva del riesgo de caída.

## Descripción general

Este repositorio contiene dos scripts principales, cada uno adaptado a una modalidad de
captura de marcha:

| Script | Modalidad | Descripción |
|---|---|---|
| `gait_analysis_offline_modo_sobresuelo5.py` | Marcha sobre suelo | Procesa ensayos de marcha overground, normalizando las series por la velocidad media de marcha. |
| `gait_analysis_offline_modo_caminadora4.py` | Marcha en caminadora | Procesa ensayos de marcha en cinta rodante, sin normalización por velocidad (velocidad constante impuesta por la banda). |

Ambos scripts comparten la misma arquitectura general (extracción de series → cálculo de
estadísticos espaciotemporales → cálculo de entropía → score clínico → score entrópico →
interpretación combinada), pero difieren en las variables disponibles y en los umbrales de
puntuación, según la modalidad de captura.

## Requisitos

- Python 3.8 o superior
- Librerías: `numpy`, `pathlib`, `os`, `shutil`, `tempfile`, `typing`, `time`, `math`
- Función de análisis de marcha externa compatible con la API de OpenCap
  (`gait_analysis_fn`), que debe exponer al menos:
  - `computeScalars(scalar_names, return_all=True)`
  - `comValues(rotate="gaitCycle", filtfreq=...)`
- Carpeta de sesión de OpenCap con la siguiente estructura esperada:
  - session_folder/
    - Marker Data/
      - <nombre_ensayo>.trc
    - OpenSimData/
      - Kinematics/
        - <nombre_ensayo>.mot
        
Cada ensayo requiere un par `.trc`/`.mot` con el mismo nombre base.

## Estructura del pipeline

1. **Detección de ensayos** (`find_trial_pairs`): busca automáticamente todos los pares
   TRC/MOT válidos dentro de la carpeta de sesión.
2. **Extracción de series** (`collect_series_from_session`): para cada ensayo y cada
   pierna (derecha/izquierda), calcula los escalares de marcha (cadencia, longitud de
   zancada/paso, velocidad, doble apoyo, etc.) y extrae la serie de aceleración
   anteroposterior (AP) del tronco/centro de masa.
3. **Estadísticos espaciotemporales** (`compute_gait_stats`): calcula medias, desviaciones
   estándar y coeficientes de variación (CV%) de las variables de marcha.
4. **Dinámica no lineal** (`compute_all_entropies`): calcula entropía muestral (SampEn) y
   entropía multiescala (MSE / AUC-MSE) sobre las series de tiempo de zancada, variabilidad
   local del tiempo de zancada, y aceleración AP del tronco.
5. **Score clínico** (`fall_risk_score`): puntúa cada variable espaciotemporal según
   umbrales basados en literatura (ver tabla de referencias más abajo), con ajuste opcional
   por historial de caídas (+15 puntos, tope 100).
6. **Score entrópico independiente** (`score_sampen_independent`, `score_aucmse_independent`):
   puntúa cada señal entrópica de forma independiente al score clínico, sin sumarse a este.
7. **Interpretación combinada** (`combined_interpretation`): genera un párrafo de
   interpretación clínica en lenguaje natural, cruzando el resultado del score clínico con el
   patrón de complejidad dinámica observado.

## Estructura del Score Clínico (0–100 pts)

| Bloque | Variable | Referencia principal |
|---|---|---|
| A | CV Tiempo de zancada | Hausdorff et al. 2001 |
| B | Velocidad de marcha / Longitud de paso* | Studenski 2011; Bohannon 1997 |
| C | Longitud de zancada/altura | Oberg 1993; Bohannon 1997 |
| D | Cadencia | Kressig 2004; Maki 1997 |
| E | CV Cadencia / Doble apoyo* | Lord et al. 2011 |
| F | CV Longitud de zancada | Hausdorff et al. 2001 |

*Las variables exactas de cada bloque varían según la modalidad (sobre suelo usa velocidad
de marcha y CV de cadencia; caminadora usa longitud de paso y doble apoyo, al no disponer de
velocidad variable).

Ajuste adicional: +15 puntos si el paciente presenta historial de caídas (Tinetti 1988),
con tope de 100 puntos. 

## Estructura del Score Entrópico (0–100 pts, independiente del score clínico)

| Métrica | Señal | Puntos máx. |
|---|---|---|
| SampEn | Tiempo de zancada | 30 |
| SampEn | Variabilidad del tiempo de zancada | 20 |
| AUC-MSE | Tiempo de zancada | 30 |
| AUC-MSE | Aceleración AP del tronco | 20 |

Cada señal se puntúa mediante una curva en "U": tanto valores excesivamente bajos
(rigidez/automatización) como excesivamente altos (caos/desorganización) respecto al rango
"Normal" se consideran patológicos. Clasificación final: **Alto**, **Moderado**, **Bajo**.

> **Nota metodológica:** los umbrales de aceleración AP del tronco en modo caminadora deben
> interpretarse con cautela, dado que la literatura reporta una reducción sistemática de la
> entropía de esta señal en modalidad caminadora respecto a sobre suelo, asociada al
> fenómeno de entrainment (sincronización sensomotora). 

## Limitaciones conocidas

- Los umbrales del Score Clínico y del Score Entrópico están basados en literatura
  heterogénea en cuanto a población, sensores y protocolo de medición; algunos (marcados como
  `provisional` o `semi_literature_based` en el código) requieren recalibración con datos
  poblacionales propios.
- El pipeline requiere una carpeta de sesión completa con pares TRC/MOT válidos; ensayos
  incompletos o corruptos se omiten automáticamente y se reportan en `errors`.
- La estimación de aceleración de tronco es una aproximación derivada de la posición del
  centro de masa (COM), no una medición directa de acelerómetro.
