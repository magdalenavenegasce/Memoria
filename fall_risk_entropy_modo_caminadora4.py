"""
fall_risk_entropy2.py — v9
────────────────────────────────────────────────────────────────────────────
SCORE CLÍNICO (0–100, bloques A–F)
────────────────────────────────────────────────────────────────────────────
Bloque A — CV Tiempo de zancada (Hausdorff et al. 2001, APMR)       máx 22 pts
> 6.0% → +22  |  4.0–6.0% → +12  |  2.5–4.0% → +5

Bloque B — Velocidad de marcha (Studenski 2011; Bohannon 1997)       máx 22 pts
< 0.6 m/s → +22  |  0.6–0.8 → +17  |  0.8–1.0 → +10  |  1.0–1.2 → +3

Bloque C — Longitud de zancada / altura (Oberg 1993; Bohannon 1997) máx 20 pts
ratio < 0.50 → +20  |  0.50–0.60 → +10  |  0.60–0.72 → +3

Bloque D — Cadencia (Kressig 2004; Maki 1997)                       máx 15 pts
< 80 spm → +15  |  80–100 → +8  |  >120 + ratio<0.72 → +5

Bloque E — CV Cadencia (Lord et al. 2011)                           máx 12 pts
> 8.0% → +12  |  5.0–8.0% → +7

Bloque F — CV Longitud de zancada (Hausdorff 2001)                  máx  9 pts
> 6.0% → +9  |  4.0–6.0% → +5

Multiplicador historial de caída (Tinetti 1988): score × 1.5 (cap 100)
Clasificación: ≥ 60 ALTO | ≥ 30 MODERADO | < 30 BAJO

────────────────────────────────────────────────────────────────────────────
SCORE ENTROPÍA INDEPENDIENTE (0–100, no se suma al score clínico)
────────────────────────────────────────────────────────────────────────────
SampEn  (0–60 pts): Tiempo de zancada máx 40 | Longitud de zancada máx 20
AUC-MSE (0–40 pts): Tiempo de zancada máx 26 | Longitud de zancada máx 14
Clasificación: ≥ 60 ALTO | ≥ 30 MODERADO | < 30 BAJO
────────────────────────────────────────────────────────────────────────────
"""

import os, pathlib, shutil, tempfile
import numpy as np
from typing import Optional
import time

# ═══════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# ═══════════════════════════════════════════════════════════

def _to_clean_array(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    return arr[np.isfinite(arr)]

def _find_trial_pairs(session_folder: str) -> list:
    session_folder = pathlib.Path(session_folder)
    marker_dir = session_folder / "MarkerData"
    kin_dir    = session_folder / "OpenSimData" / "Kinematics"
    if not marker_dir.is_dir():
        raise FileNotFoundError(f"No se encontró MarkerData en: {session_folder}")
    if not kin_dir.is_dir():
        raise FileNotFoundError(f"No se encontró OpenSimData/Kinematics en: {session_folder}")
    pairs = []
    for trc_file in sorted(marker_dir.glob("*.trc")):
        mot_file = kin_dir / f"{trc_file.stem}.mot"
        if mot_file.is_file():
            pairs.append({"name": trc_file.stem,
                          "trc":  str(trc_file),
                          "mot":  str(mot_file)})
    return pairs

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def extract_trunk_ap_acceleration_series(ga, filt_freq: int = 10) -> np.ndarray:
    """
    Extrae la aceleración AP del COM en gait frame.
    Se usa como aproximación de aceleración AP del tronco.
    """
    try:
        com_vals = ga.comValues(rotate='gaitCycle', filt_freq=filt_freq)
    except TypeError:
        com_vals = ga.comValues(rotate='gaitCycle')

    if com_vals is None or len(com_vals) < 5:
        return np.array([])

    cols = list(com_vals.columns)
    if "time" not in cols or "x" not in cols:
        return np.array([])

    t = _to_clean_array(com_vals["time"].to_numpy())
    x = _to_clean_array(com_vals["x"].to_numpy())

    n = min(len(t), len(x))
    t = t[:n]
    x = x[:n]

    if n < 5:
        return np.array([])

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.array([])

    dt_med = float(np.median(dt))

    v = np.gradient(x, dt_med)
    a = np.gradient(v, dt_med)

    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]

    return a

# ═══════════════════════════════════════════════════════════
# 1. ENTROPÍA MUESTRAL (SampEn)
# ═══════════════════════════════════════════════════════════
def sample_entropy(time_series: np.ndarray,
                   m: int = 2,
                   r: Optional[float] = None) -> float:
    """
    SampEn con menos overhead de memoria.
    Mantiene exactamente la definición estándar.
    """
    x = _to_clean_array(time_series)
    N = len(x)

    if N < 10:
        return np.nan

    if r is None:
        sd = np.std(x, ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            return np.nan
        r = 0.2 * sd

    if not np.isfinite(r) or r <= 0:
        return np.nan

    def _count_matches(mm: int) -> float:
        ntemp = N - mm + 1
        if ntemp < 2:
            return 0.0

        count = 0
        for i in range(ntemp - 1):
            ref = x[i:i + mm]
            maxdist = np.zeros(ntemp - i - 1, dtype=float)

            for k in range(mm):
                diffs = np.abs(x[i + 1 + k:ntemp + k] - ref[k])
                maxdist = np.maximum(maxdist, diffs)

            count += np.sum(maxdist < r)

        return float(count)

    B = _count_matches(m)
    A = _count_matches(m + 1)

    if A <= 0 or B <= 0:
        return np.nan

    return float(-np.log(A / B))

# ═══════════════════════════════════════════════════════════
# 2. ENTROPÍA MULTIESCALA (MSE)
# ═══════════════════════════════════════════════════════════
def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    N = len(x)
    t = x[:N - (N % scale)] if N % scale != 0 else x
    return np.mean(t.reshape(-1, scale), axis=1)


def multiscale_entropy(time_series: np.ndarray,
                       max_scale: int = 5,
                       m: int = 2,
                       r: Optional[float] = None) -> dict:
    """
    MSE sobre múltiples escalas.
    Mantiene la lógica original, pero con algo más de control y robustez.
    """
    x = _to_clean_array(time_series)
    N = len(x)


    if N < 10:
        return {
            "by_scale": {s: np.nan for s in range(1, max_scale + 1)},
            "auc_mse": np.nan
        }


    effective_max = min(max_scale, max(1, N // 8))


    if r is None:
        sd = np.std(x, ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            return {
                "by_scale": {s: np.nan for s in range(1, max_scale + 1)},
                "auc_mse": np.nan
            }
        r = 0.2 * sd


    vals = {}
    for s in range(1, effective_max + 1):
        cg = _coarse_grain(x, s)
        vals[s] = sample_entropy(cg, m=m, r=r) if len(cg) >= 10 else np.nan


    for s in range(effective_max + 1, max_scale + 1):
        vals[s] = np.nan


    valid_scales = [s for s, v in vals.items() if np.isfinite(v)]
    valid_vals = [vals[s] for s in valid_scales]


    if len(valid_vals) >= 2:
        raw_auc = float(np.trapz(valid_vals, x=valid_scales))
        scale_range = valid_scales[-1] - valid_scales[0]
        auc = raw_auc / scale_range if scale_range > 0 else np.nan
    else:
        auc = np.nan


    return {
        "by_scale": vals,
        "auc_mse": auc
    }

# ═══════════════════════════════════════════════════════════
# 3. ENTROPÍAS PARA TODAS LAS VARIABLES
# ═══════════════════════════════════════════════════════════
def compute_all_entropies(stride_times: np.ndarray,
                          trunk_ap_acc: Optional[np.ndarray] = None,
                          max_scale: int = 5) -> dict:
    out = {}

    out["stride_time"] = compute_stride_time_entropy(
        stride_times=stride_times,
        max_scale=max_scale,
        m=2,
        r=None
    )

    out["stride_time_variability"] = compute_stride_time_variability_entropy(
        stride_times=stride_times,
        window=5,
        m=2,
        r=None
    )

    out["trunk_ap_acceleration"] = compute_trunk_ap_mse(
        trunk_ap_acc=trunk_ap_acc,
        max_scale=max_scale,
        m=2,
        r=None
    )

    return out

# ═══════════════════════════════════════════════════════════
# 3A. ENTROPÍA ESPECÍFICA DE TIEMPO DE ZANCADA
# ═══════════════════════════════════════════════════════════
def compute_stride_time_entropy(stride_times: np.ndarray,
                                max_scale: int = 5,
                                m: int = 2,
                                r: Optional[float] = None) -> dict:
    """
    Calcula SampEn y MSE para la serie cruda de tiempo de zancada,
    sin normalización por velocidad.
    """
    st_raw = _to_clean_array(stride_times)

    if st_raw.size < 10:
        return {
            "sampen": np.nan,
            "mse": {"by_scale": {s: np.nan for s in range(1, max_scale + 1)},
                    "auc_mse": np.nan},
            "n": int(st_raw.size),
            "mean_raw": float(np.mean(st_raw)) if st_raw.size else np.nan,
            "sd_raw": float(np.std(st_raw, ddof=1)) if st_raw.size >= 2 else np.nan,
            "cv_pct_raw": (float(np.std(st_raw, ddof=1)) / float(np.mean(st_raw)) * 100.0)
                          if st_raw.size >= 2 and np.mean(st_raw) > 0 else np.nan,
            "mean_norm": float(np.mean(st_raw)) if st_raw.size else np.nan,
            "sd_norm": float(np.std(st_raw, ddof=1)) if st_raw.size >= 2 else np.nan,
            "speed_mean": np.nan,
            "normalized_by_speed": False,
            "r_used": np.nan
        }

    sd_raw = float(np.std(st_raw, ddof=1))
    mean_raw = float(np.mean(st_raw))
    r_used = float(0.2 * sd_raw) if r is None else float(r)

    return {
        "sampen": sample_entropy(st_raw, m=m, r=r_used),
        "mse": multiscale_entropy(st_raw, max_scale=max_scale, m=m, r=r_used),
        "n": int(st_raw.size),
        "mean_raw": mean_raw,
        "sd_raw": sd_raw,
        "cv_pct_raw": (sd_raw / mean_raw * 100.0) if mean_raw > 0 else np.nan,
        "mean_norm": mean_raw,
        "sd_norm": sd_raw,
        "speed_mean": np.nan,
        "normalized_by_speed": False,
        "r_used": r_used
    }

# ═══════════════════════════════════════════════════════════
# 3B. ENTROPÍA ESPECÍFICA DE VARIABILIDAD DEL TIEMPO DE ZANCADA
# ═══════════════════════════════════════════════════════════
def build_stride_time_variability_series(stride_times: np.ndarray,
                                         window: int = 5) -> dict:
    """
    Construye una serie de variabilidad local del tiempo de zancada
    usando CV rolling (%), sin normalización por velocidad.
    """
    st = _to_clean_array(stride_times)

    if st.size < max(window + 2, 6):
        return {
            "raw_series": np.array([]),
            "norm_series": np.array([]),
            "speed_mean": np.nan,
            "window": window,
            "n": 0,
            "normalized_by_speed": False
        }

    raw_series = []
    for i in range(len(st) - window + 1):
        seg = st[i:i + window]
        mu = float(np.mean(seg))
        sd = float(np.std(seg, ddof=1))
        if mu > 0 and np.isfinite(mu) and np.isfinite(sd):
            raw_series.append((sd / mu) * 100.0)

    raw_series = np.asarray(raw_series, dtype=float)
    raw_series = raw_series[np.isfinite(raw_series)]
    norm_series = raw_series.copy()

    return {
        "raw_series": raw_series,
        "norm_series": norm_series,
        "speed_mean": np.nan,
        "window": window,
        "n": int(norm_series.size),
        "normalized_by_speed": False
    }


def compute_stride_time_variability_entropy(stride_times: np.ndarray,
                                            window: int = 5,
                                            m: int = 2,
                                            r: Optional[float] = None) -> dict:
    """
    Calcula SampEn de la variabilidad local del tiempo de zancada
    (serie rolling de CV), sin normalización por velocidad.
    """
    var_data = build_stride_time_variability_series(
        stride_times=stride_times,
        window=window
    )

    raw_series = var_data["raw_series"]
    norm_series = var_data["norm_series"]

    if norm_series.size < 10:
        return {
            "sampen": np.nan,
            "n": int(norm_series.size),
            "window": window,
            "series_mean_raw": float(np.mean(raw_series)) if raw_series.size else np.nan,
            "series_sd_raw": float(np.std(raw_series, ddof=1)) if raw_series.size >= 2 else np.nan,
            "series_mean_norm": float(np.mean(norm_series)) if norm_series.size else np.nan,
            "series_sd_norm": float(np.std(norm_series, ddof=1)) if norm_series.size >= 2 else np.nan,
            "speed_mean": np.nan,
            "normalized_by_speed": False,
            "r_used": np.nan
        }

    sd_norm = float(np.std(norm_series, ddof=1))
    r_used = float(0.2 * sd_norm) if r is None else float(r)

    return {
        "sampen": sample_entropy(norm_series, m=m, r=r_used),
        "n": int(norm_series.size),
        "window": window,
        "series_mean_raw": float(np.mean(raw_series)),
        "series_sd_raw": float(np.std(raw_series, ddof=1)) if raw_series.size >= 2 else np.nan,
        "series_mean_norm": float(np.mean(norm_series)),
        "series_sd_norm": sd_norm,
        "speed_mean": np.nan,
        "normalized_by_speed": False,
        "r_used": r_used
    }

# ═══════════════════════════════════════════════════════════
# 3C. ENTROPÍA ESPECÍFICA DE ACELERACIÓN AP DEL TRONCO/COM
# ═══════════════════════════════════════════════════════════
def compute_trunk_ap_mse(
    trunk_ap_acc: np.ndarray,
    max_scale: int = 15,
    m: int = 2,
    r: Optional[float] = None,
    target_fs_ratio: int = 4,
) -> dict:
    """Calcula MSE para aceleración AP del tronco/COM.

    Aplica estandarización (Z-score) y submuestreo previo para ajustar señales
    continuas sobremuestreadas.
    """
    acc_raw = _to_clean_array(trunk_ap_acc)

    if acc_raw.size == 0:
        return {
            "mse": {
                "by_scale": {s: np.nan for s in range(1, max_scale + 1)},
                "auc_mse": np.nan,
            },
            "n": 0,
            "mean_raw": np.nan,
            "sd_raw": np.nan,
            "mean_norm": np.nan,
            "sd_norm": np.nan,
            "speed_mean": np.nan,
            "normalized_by_speed": False,
            "r_used": np.nan,
        }

    if acc_raw.size < 10:
        return {
            "mse": {
                "by_scale": {s: np.nan for s in range(1, max_scale + 1)},
                "auc_mse": np.nan,
            },
            "n": int(acc_raw.size),
            "mean_raw": float(np.mean(acc_raw)),
            "sd_raw": (
                float(np.std(acc_raw, ddof=1)) if acc_raw.size >= 2 else np.nan
            ),
            "mean_norm": float(np.mean(acc_raw)),
            "sd_norm": (
                float(np.std(acc_raw, ddof=1)) if acc_raw.size >= 2 else np.nan
            ),
            "speed_mean": np.nan,
            "normalized_by_speed": False,
            "r_used": np.nan,
        }

    sd_raw = float(np.std(acc_raw, ddof=1))

    # 1. SUBMUESTREO (Downsampling) para reducir sobremuestreo si la señal es muy larga (> 5000 puntos)
    if acc_raw.size > 5000:
        acc_processed = acc_raw[::target_fs_ratio]
    else:
        acc_processed = acc_raw

    # 2. ESTANDARIZACIÓN Z-SCORE (Media = 0, SD = 1)
    sd_proc = np.std(acc_processed, ddof=1)
    if sd_proc > 0:
        z_acc = (acc_processed - np.mean(acc_processed)) / sd_proc
    else:
        z_acc = acc_processed - np.mean(acc_processed)

    # Definir tolerancia r (si es None, se calcula sobre la señal estandarizada = 0.2 * 1.0)
    r_used = float(0.2 * np.std(z_acc, ddof=1)) if r is None else float(r)

    # 3. CÁLCULO DE MSE SOBRE LA SEÑAL TRATADA
    t0 = time.perf_counter()
    mse_result = multiscale_entropy(
        z_acc, max_scale=max_scale, m=m, r=r_used
    )
    t1 = time.perf_counter()

    print(
        f"[DEBUG trunk] n_raw={acc_raw.size} -> n_proc={z_acc.size} | MSE"
        f" time={t1 - t0:.4f}s"
    )

    return {
        "mse": mse_result,
        "n": int(acc_raw.size),
        "mean_raw": float(np.mean(acc_raw)),
        "sd_raw": sd_raw,
        "mean_norm": float(np.mean(z_acc)),
        "sd_norm": float(np.std(z_acc, ddof=1)),
        "speed_mean": np.nan,
        "normalized_by_speed": False,
        "r_used": r_used,
    }

# ═══════════════════════════════════════════════════════════
# 4. ESTADÍSTICOS DE MARCHA
# ═══════════════════════════════════════════════════════════
def compute_gait_stats(stride_times: np.ndarray,
                       step_lengths: Optional[np.ndarray] = None,
                       cadences: Optional[np.ndarray] = None,
                       double_support_times: Optional[np.ndarray] = None) -> dict:
    out = {}

    stride_times = _to_clean_array(stride_times)
    if stride_times.size > 0:
        out["stride_time_mean_s"] = float(np.mean(stride_times))
        out["stride_time_sd_s"] = float(np.std(stride_times, ddof=1)) if stride_times.size >= 2 else np.nan
        out["stride_time_cv_pct"] = (
            out["stride_time_sd_s"] / out["stride_time_mean_s"] * 100.0
            if out["stride_time_mean_s"] != 0 and np.isfinite(out["stride_time_sd_s"])
            else np.nan
        )

    if step_lengths is not None:
        step_lengths = _to_clean_array(step_lengths)
        if step_lengths.size > 0:
            out["step_length_mean_m"] = float(np.mean(step_lengths))
            out["step_length_sd_m"] = float(np.std(step_lengths, ddof=1)) if step_lengths.size >= 2 else np.nan
            out["step_length_cv_pct"] = (
                out["step_length_sd_m"] / out["step_length_mean_m"] * 100.0
                if out["step_length_mean_m"] != 0 and np.isfinite(out["step_length_sd_m"])
                else np.nan
            )

    if cadences is not None:
        cadences = _to_clean_array(cadences)
        cadences = cadences[cadences > 0]
        if cadences.size > 0:
            out["cadence_mean_spm"] = float(np.mean(cadences))
            out["cadence_sd_spm"] = float(np.std(cadences, ddof=1)) if cadences.size >= 2 else np.nan
            out["cadence_cv_pct"] = (
                out["cadence_sd_spm"] / out["cadence_mean_spm"] * 100.0
                if out["cadence_mean_spm"] != 0 and np.isfinite(out["cadence_sd_spm"])
                else np.nan
            )

    if double_support_times is not None:
        double_support_times = _to_clean_array(double_support_times)
        if double_support_times.size > 0:
            out["double_support_mean_pct"] = float(np.mean(double_support_times))
            out["double_support_sd_pct"] = float(np.std(double_support_times, ddof=1)) if double_support_times.size >= 2 else np.nan
            out["double_support_cv_pct"] = (
                out["double_support_sd_pct"] / out["double_support_mean_pct"] * 100.0
                if out["double_support_mean_pct"] != 0 and np.isfinite(out["double_support_sd_pct"])
                else np.nan
            )

    return out
 
# ═══════════════════════════════════════════════════════════
# 5. SCORE NUMÉRICO ACUMULADO
# ═══════════════════════════════════════════════════════════
def score_stride_time_mean_treadmill(st_mean_s: float) -> tuple:
    """
    Tiempo de zancada medio.
    Variable de apoyo, con menor peso clínico que la variabilidad temporal.
    Rango de referencia práctico en adultos mayores a velocidad confortable:
    ~1.0 a 1.3 s suele ser razonable; valores claramente más lentos sugieren marcha cautelosa.
    """
    if np.isnan(st_mean_s):
        return 0, None

    if st_mean_s > 1.30:
        return 10, f"Tiempo de zancada aumentado ({st_mean_s:.3f} s > 1.30 s)"
    elif st_mean_s > 1.20:
        return 5, f"Tiempo de zancada limítrofe-alto ({st_mean_s:.3f} s, 1.21–1.30 s)"
    elif st_mean_s < 0.90:
        return 5, f"Tiempo de zancada limítrofe-bajo ({st_mean_s:.3f} s < 0.90 s)"
    else:
        return 0, None


def score_stride_time_cv_treadmill(cv_pct: float) -> tuple:
    """
    CV del tiempo de zancada.
    Variable principal del score por su mejor relación con riesgo de caída.
    Se usan cortes clínicos pragmáticos con mayor penalización al aumento claro de variabilidad.
    """
    if np.isnan(cv_pct):
        return 0, None

    if cv_pct >= 6.0:
        return 35, f"Variabilidad del tiempo de zancada marcadamente aumentada (CV {cv_pct:.2f}% >= 6.0%)"
    elif cv_pct >= 4.0:
        return 20, f"Variabilidad del tiempo de zancada aumentada (CV {cv_pct:.2f}% entre 4.0% y 5.9%)"
    elif cv_pct >= 2.5:
        return 8, f"Variabilidad del tiempo de zancada limítrofe (CV {cv_pct:.2f}% entre 2.5% y 3.9%)"
    else:
        return 0, None


def score_step_length_treadmill(step_len_m: float, height: float = 0.0) -> tuple:
    """
    Longitud de paso media.
    Se prefiere normalizar por talla si está disponible.
    Penalización moderada: útil clínicamente, pero menos robusta que la variabilidad.
    """
    if np.isnan(step_len_m):
        return 0, None

    if height > 0:
        ratio = step_len_m / height
        if ratio < 0.28:
            return 15, f"Longitud de paso reducida ({step_len_m:.3f} m; ratio talla {ratio:.3f} < 0.28)"
        elif ratio < 0.32:
            return 8, f"Longitud de paso limítrofe-baja ({step_len_m:.3f} m; ratio talla {ratio:.3f} entre 0.28 y 0.32)"
        else:
            return 0, None
    else:
        if step_len_m < 0.45:
            return 15, f"Longitud de paso reducida ({step_len_m:.3f} m < 0.45 m)"
        elif step_len_m < 0.55:
            return 8, f"Longitud de paso limítrofe-baja ({step_len_m:.3f} m entre 0.45 y 0.55 m)"
        else:
            return 0, None


def score_cadence_treadmill(cad_spm: float) -> tuple:
    """
    Cadencia media.
    Penalización moderada; marcador de ritmo, menos robusto que la variabilidad.
    """
    if np.isnan(cad_spm):
        return 0, None
    if cad_spm < 90:
        return 15, f"Cadencia reducida ({cad_spm:.1f} pasos/min < 90)"
    elif cad_spm < 100:
        return 8, f"Cadencia limítrofe-baja ({cad_spm:.1f} pasos/min entre 90 y 100)"
    return 0, None

def score_double_support_treadmill(ds_pct: float) -> tuple:
    """
    Porcentaje de doble apoyo.
    Variable secundaria fuerte en la literatura prospectiva.
    """
    if np.isnan(ds_pct):
        return 0, None

    if ds_pct >= 28.0:
        return 25, f"Doble apoyo aumentado ({ds_pct:.1f}% >= 28%)"
    elif ds_pct >= 24.0:
        return 15, f"Doble apoyo limítrofe-alto ({ds_pct:.1f}% entre 24% y 27.9%)"
    elif ds_pct >= 20.0:
        return 6, f"Doble apoyo discretamente elevado ({ds_pct:.1f}% entre 20% y 23.9%)"
    else:
        return 0, None

def _score_sampen_independent(entropies: dict) -> dict:
    """
    Score SampEn independiente 0-50 pts. Curva en U por señal.

    Señales:
    - Tiempo de zancada: máx 30 pts
    - Variabilidad del tiempo de zancada: máx 20 pts

    Rangos:
    Tiempo de zancada (basado en la tabla referencial acordada)
    - Normal:      1.20–1.50
    - Limítrofe:   1.00–1.20 / 1.50–1.70
    - Patológico:  <1.00 o >1.70

    Variabilidad local del tiempo de zancada
    - Se mantiene provisional mientras no exista una tabla específica
      basada en media±SD para esta señal derivada.
    - Normal:      1.20–2.00
    - Limítrofe:   1.00–1.20 / 2.00–2.50
    - Patológico:  <1.00 o >2.50
    """
    _sigs = [
        ("stride_time", 30, 15, 0.80, 1.10, 0.65, 1.25, True, "literature_synthesis"),
        ("stride_time_variability", 20, 10, 1.00, 2.50, 0.50, 4.00, False, "provisional"),
    ]

    total = 0
    breakdown = []

    for sig, p_pat, p_lim, nlo, nhi, blo, bhi, lit_based, status in _sigs:
        se = entropies.get(sig, {}).get("sampen", np.nan)

        if np.isnan(se):
            pts = 0
            categoria = "N/D"
        elif se < blo or se > bhi:
            pts = p_pat
            categoria = "Patológico"
        elif (blo <= se < nlo) or (nhi < se <= bhi):
            pts = p_lim
            categoria = "Limítrofe"
        else:
            pts = 0
            categoria = "Normal"

        total += pts
        breakdown.append({
            "signal": sig,
            "sampen": se,
            "pts": pts,
            "categoria": categoria,
            "pts_max": p_pat,
            "normal_range": (nlo, nhi),
            "borderline_range": ((blo, nlo), (nhi, bhi)),
            "pathological_rule": f"<{blo:.2f} or >{bhi:.2f}",
            "literature_based": lit_based,
            "reference_population": "older_adults",
            "status": status
        })

    return {
        "score": total,
        "score_max": 50,
        "breakdown": breakdown
    }

def _score_aucmse_independent(entropies: dict) -> dict:
    """
    Score AUC-MSE independiente 0-50 pts. Curva en U por señal.

    Señales:
    - Tiempo de zancada: máx 30 pts
    - Aceleración AP del tronco: máx 20 pts

    Rangos:
    Tiempo de zancada
    - Normal:      0.92–1.15
    - Limítrofe:   0.82–0.92 / 1.15–1.28
    - Patológico:  <0.82 o >1.28

    Aceleración AP del tronco
    - Normal:      0.88–1.12
    - Limítrofe:   0.78–0.88 / 1.12–1.25
    - Patológico:  <0.78 o >1.25

    Nota:
    - trunk_ap_acceleration está alineada con la tabla conceptual actual.
    - Estos cortes siguen siendo operativos/provisionales hasta consolidar
      una tabla media±SD específica para adultos mayores.
    """
    _sigs = [
        ("stride_time", 30, 15, 0.80, 1.00, 0.70, 1.15, False, "provisional"),
        ("trunk_ap_acceleration", 20, 10, 0.75, 0.90, 0.65, 1.05, True, "semi_literature_based"),
    ]

    total = 0
    breakdown = []

    for sig, p_pat, p_lim, nlo, nhi, blo, bhi, lit_based, status in _sigs:
        auc = entropies.get(sig, {}).get("mse", {}).get("auc_mse", np.nan)

        if np.isnan(auc):
            pts = 0
            categoria = "N/D"
        elif auc < blo or auc > bhi:
            pts = p_pat
            categoria = "Patológico"
        elif (blo <= auc < nlo) or (nhi < auc <= bhi):
            pts = p_lim
            categoria = "Limítrofe"
        else:
            pts = 0
            categoria = "Normal"

        total += pts
        breakdown.append({
            "signal": sig,
            "auc_mse": auc,
            "pts": pts,
            "categoria": categoria,
            "pts_max": p_pat,
            "normal_range": (nlo, nhi),
            "borderline_range": ((blo, nlo), (nhi, bhi)),
            "pathological_rule": f"<{blo:.2f} or >{bhi:.2f}",
            "literature_based": lit_based,
            "reference_population": "older_adults",
            "status": status
        })

    return {
        "score": total,
        "score_max": 50,
        "breakdown": breakdown
    }

def combined_interpretation(score_base: float,
                            risk_level: str,
                            score_sampen: float,
                            score_aucmse: float,
                            entropies: dict) -> str:
    """
    Genera un párrafo de interpretación clínica combinando el score clínico
    de caminadora con los scores de entropía SampEn y AUC-MSE.
    Adaptado al modelo nuevo:
    - tiempo de zancada / su variabilidad
    - longitud de paso
    - cadencia
    - doble apoyo
    """

    import math

    def safe(v):
        try:
            fv = float(v)
            return fv if not math.isnan(fv) else None
        except Exception:
            return None

    sb = safe(score_base)
    sse = safe(score_sampen)
    sauc = safe(score_aucmse)
    lv = str(risk_level).upper()

    sestr = f"{sse:.0f}/50" if sse is not None else "ND"
    aucstr = f"{sauc:.0f}/50" if sauc is not None else "ND"

    if sse is None:
        patron = "indeterminado"
    elif sse >= 40:
        patron = "muy alterado"
    elif sse >= 20:
        patron = "intermedio"
    else:
        patron = "conservado"

    if lv == "ALTO" and patron == "muy alterado":
        parrafo = (
            f"El score clínico de caminadora indica riesgo ALTO de caída. "
            f"Este resultado es compatible con alteraciones relevantes en variables temporales y espaciales de la marcha, "
            f"incluyendo la longitud de paso, la cadencia y/o el tiempo en doble apoyo. "
            f"Además, la complejidad dinámica se encuentra claramente alterada "
            f"(Score SampEn {sestr}), lo que sugiere un patrón locomotor menos estable, menos adaptable y más vulnerable a perturbaciones."
        )
    elif lv == "ALTO" and patron == "intermedio":
        parrafo = (
            f"El score clínico de caminadora indica riesgo ALTO de caída. "
            f"Las alteraciones promedio de la marcha tienen suficiente magnitud clínica como para elevar el riesgo, "
            f"y la entropía muestra señales adicionales de compromiso dinámico "
            f"(Score SampEn {sestr}). "
            f"Este perfil puede corresponder a una marcha ineficiente o cautelosa, con capacidad de adaptación reducida."
        )
    elif lv == "ALTO" and patron == "conservado":
        parrafo = (
            f"El score clínico de caminadora indica riesgo ALTO de caída. "
            f"En este caso, la complejidad dinámica global se mantiene relativamente conservada "
            f"(Score SampEn {sestr}; Score AUC-MSE {aucstr}), "
            f"por lo que el riesgo parece estar impulsado principalmente por alteraciones clínicas en la longitud de paso, "
            f"la cadencia, el tiempo de zancada y/o el porcentaje de doble apoyo. "
            f"Este perfil es compatible con una estrategia de marcha cautelosa, rígida o mecánicamente ineficiente."
        )
    elif lv == "MODERADO" and patron == "muy alterado":
        parrafo = (
            f"El score clínico de caminadora indica riesgo MODERADO de caída. "
            f"Aunque las alteraciones clínicas promedio no alcanzan el nivel más severo, "
            f"la complejidad dinámica está claramente comprometida "
            f"(Score SampEn {sestr}), "
            f"lo que añade una señal importante de inestabilidad locomotora. "
            f"Este patrón puede reflejar una marcha menos robusta frente a cambios del entorno o demandas motoras adicionales."
        )
    elif lv == "MODERADO" and patron == "intermedio":
        parrafo = (
            f"El score clínico de caminadora indica riesgo MODERADO de caída. "
            f"Se observan desviaciones parciales en parámetros promedio de la marcha y, al mismo tiempo, "
            f"la complejidad dinámica muestra compromiso leve a moderado "
            f"(Score SampEn {sestr}). "
            f"En conjunto, este perfil sugiere una marcha con eficiencia reducida y con cierto deterioro en la capacidad de adaptación."
        )
    elif lv == "MODERADO" and patron == "conservado":
        parrafo = (
            f"El score clínico de caminadora indica riesgo MODERADO de caída. "
            f"La complejidad dinámica se mantiene sin señales mayores de alarma "
            f"(Score SampEn {sestr}; Score AUC-MSE {aucstr}), "
            f"de modo que el compromiso parece concentrarse sobre todo en los promedios espaciotemporales, "
            f"como la longitud de paso, la cadencia, el tiempo de zancada o el doble apoyo. "
            f"Este perfil es compatible con una marcha cautelosa o levemente ineficiente que requiere seguimiento."
        )
    elif lv == "BAJO" and patron == "muy alterado":
        parrafo = (
            f"El score clínico de caminadora indica riesgo BAJO de caída según los parámetros promedio analizados "
            f"(score clínico {sb:.0f}/100)." if sb is not None else
            f"El score clínico de caminadora indica riesgo BAJO de caída según los parámetros promedio analizados. "
        )
        parrafo += (
            f" Sin embargo, la complejidad dinámica aparece alterada "
            f"(Score SampEn {sestr}), "
            f"por lo que conviene interpretar el resultado con cautela. "
            f"Esto podría indicar cambios tempranos del control locomotor que no se reflejan aún de forma marcada en los promedios de marcha."
        )
    elif lv == "BAJO":
        parrafo = (
            f"El score clínico de caminadora indica riesgo BAJO de caída "
            f"(score clínico {sb:.0f}/100)." if sb is not None else
            f"El score clínico de caminadora indica riesgo BAJO de caída. "
        )
        parrafo += (
            f" En conjunto, la longitud de paso, la cadencia, el tiempo de zancada y el doble apoyo "
            f"se mantienen dentro de rangos clínicamente aceptables o con desviaciones menores. "
            f"La complejidad dinámica también es adecuada "
            f"(Score SampEn {sestr}; Score AUC-MSE {aucstr}), "
            f"sin señales claras de desorganización locomotora."
        )
    else:
        parrafo = (
            f"Score clínico de caminadora "
            f"{f'{sb:.0f}/100' if sb is not None else 'ND'}. "
            f"Nivel de riesgo: {lv}. "
            f"Score SampEn: {sestr}. "
            f"Score AUC-MSE: {aucstr}. "
            f"Interpretación combinada no disponible."
        )

    return parrafo

def fall_risk_score(gait_stats: dict,
                  entropies: dict,
                  fall_history: bool = False,
                  height: float = 0.0) -> dict:
    """
    Score clínico para CAMINADORA basado en:
    - CV del tiempo de zancada
    - longitud de paso media
    - cadencia media
    - doble apoyo medio

    Total clínico: 0-100
    Ajuste por historial de caída: +15 puntos (cap 100)

    Clasificación:
    - >= 50: ALTO
    - >= 25: MODERADO
    - < 25: BAJO
    """

    score = 0.0
    factors = []

    st_cv = gait_stats.get("stride_time_cv_pct", np.nan)
    step = gait_stats.get("step_length_mean_m", np.nan)
    cad = gait_stats.get("cadence_mean_spm", np.nan)
    ds = gait_stats.get("double_support_mean_pct", np.nan)

    def score_stridetime_cv_treadmill(cvpct: float):
        if np.isnan(cvpct):
            return 0, None
        if cvpct >= 6.0:
            return 30, f"Variabilidad del tiempo de zancada aumentada ({cvpct:.1f}%)"
        elif cvpct >= 4.0:
            return 15, f"Variabilidad del tiempo de zancada limítrofe ({cvpct:.1f}%)"
        elif cvpct >= 2.5:
            return 5, f"Variabilidad del tiempo de zancada discretamente elevada ({cvpct:.1f}%)"
        return 0, None

    def score_step_length_treadmill(stepm: float, patient_height: float = 0.0):
        if np.isnan(stepm):
            return 0, None

        if patient_height and patient_height > 0:
            ratio = stepm / patient_height
            if ratio < 0.28:
                return 25, f"Longitud de paso reducida ({stepm:.3f} m; razón talla {ratio:.3f})"
            elif ratio < 0.33:
                return 12, f"Longitud de paso limítrofe-baja ({stepm:.3f} m; razón talla {ratio:.3f})"
            elif ratio < 0.37:
                return 5, f"Longitud de paso discretamente baja ({stepm:.3f} m; razón talla {ratio:.3f})"
            return 0, None
        else:
            if stepm < 0.45:
                return 25, f"Longitud de paso reducida ({stepm:.3f} m)"
            elif stepm < 0.55:
                return 12, f"Longitud de paso limítrofe-baja ({stepm:.3f} m)"
            elif stepm < 0.65:
                return 5, f"Longitud de paso discretamente baja ({stepm:.3f} m)"
            return 0, None

    pts, msg = score_stridetime_cv_treadmill(st_cv)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque A - CV Tiempo de zancada] valor={st_cv:.2f} -> pts={pts}")

    pts, msg = score_step_length_treadmill(step, height)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque B - Longitud de paso] valor={step:.2f} -> pts={pts}")   

    pts, msg = score_cadence_treadmill(cad)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque C - Cadencia] valor={cad:.2f} -> pts={pts}")

    pts, msg = score_double_support_treadmill(ds)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque D - Doble apoyo] valor={ds:.2f} -> pts={pts}")

    se_result = _score_sampen_independent(entropies)
    auc_result = _score_aucmse_independent(entropies)

    score_sampen = se_result["score"]
    score_auc_mse = auc_result["score"]
    score_entropy_total = score_sampen + score_auc_mse

    if score_entropy_total >= 50:
        entropy_level = "ALTO"
    elif score_entropy_total >= 25:
        entropy_level = "MODERADO"
    else:
        entropy_level = "BAJO"

    score_base = round(score, 1)

    if fall_history:
        score = min(score + 15.0, 100.0)

    score = round(score, 1)

    if score >= 50:
        level = "ALTO"
    elif score >= 25:
        level = "MODERADO"
    else:
        level = "BAJO"

    entropy_note_parts = []
    for bd in se_result["breakdown"]:
        sig_label = bd["signal"].replace("_", " ").title()
        if not np.isnan(bd["sampen"]):
            entropy_note_parts.append(
                f"SampEn {sig_label}: {bd['sampen']:.3f} ({bd['categoria']}, {bd['pts']} pts)"
            )

    for bd in auc_result["breakdown"]:
        sig_label = bd["signal"].replace("_", " ").title()
        if not np.isnan(bd["auc_mse"]):
            entropy_note_parts.append(
                f"AUC-MSE {sig_label}: {bd['auc_mse']:.3f} ({bd['categoria']}, {bd['pts']} pts)"
            )

    if fall_history:
        interpretation_txt = (
            f"Score clínico de caminadora basado en variabilidad del tiempo de zancada, "
            f"longitud de paso, cadencia y porcentaje de doble apoyo. "
            f"Se aplicó ajuste por historial de caída (+15 pts); score base {score_base}."
        )
    else:
        interpretation_txt = (
            f"Score clínico de caminadora basado en variabilidad del tiempo de zancada, "
            f"longitud de paso, cadencia y porcentaje de doble apoyo. "
            f"No se aplicó ajuste por historial de caída."
        )

    return {
        "score_base": score_base,
        "score_final": score,
        "score_sampen": score_sampen,
        "score_auc_mse": score_auc_mse,
        "score_entropy_total": score_entropy_total,
        "_entropy_level": entropy_level,
        "entropy_sampen_breakdown": se_result["breakdown"],
        "entropy_auc_mse_breakdown": auc_result["breakdown"],
        "fall_history_applied": fall_history,
        "risk_level": level,
        "global_risk_level": level,
        "contributing_factors": factors if factors else ["Sin factores clínicos principales detectados en caminadora"],
        "_entropy_note": " | ".join(entropy_note_parts) if entropy_note_parts else "Sin datos de entropía",
        "interpretation": interpretation_txt,
        "combined_interpretation": combined_interpretation(
            score_base, level, score_sampen, score_auc_mse, entropies
        )
    }

# ═══════════════════════════════════════════════════════════
# 6. RECOLECTAR SERIES DE TODOS LOS ENSAYOS
# ═══════════════════════════════════════════════════════════
def _extract_step_length_array(raw) -> np.ndarray:
    if isinstance(raw, dict):
        if not raw:
            return np.array([])
        return np.concatenate([_to_clean_array(v) for v in raw.values()])
    if raw is None:
        return np.array([])
    return _to_clean_array(raw)


def collect_series_from_session(session_folder: str,
                                 gait_analysis_fn,
                                 filter_frequency: int = 6,
                                 trim_start: float = 0.05,
                                 trim_end:   float = 0.05,
                                 verbose: bool = True) -> dict:
    scalar_names = {
        "cadence",
        "step_length",
        "double_support_time",
    }
    pairs = _find_trial_pairs(session_folder)
    if not pairs:
        raise FileNotFoundError(f"No se encontraron pares TRC+MOT en: {session_folder}")

    all_cad, all_step_length, all_ds, all_trunk_ap_acc = [], [], [], []
    processed, errors = [], []

    for pair in pairs:
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="OpenCapOffline_ent_"))
        try:
            (tmp_dir / "MarkerData").mkdir(parents=True)
            (tmp_dir / "OpenSimData" / "Kinematics").mkdir(parents=True)
            shutil.copy2(pair["trc"], str(tmp_dir / "MarkerData" / f"{pair['name']}.trc"))
            shutil.copy2(pair["mot"], str(tmp_dir / "OpenSimData" / "Kinematics" / f"{pair['name']}.mot"))

            for leg in ("r", "l"):
                ga = gait_analysis_fn(
                    str(tmp_dir), pair["name"], leg=leg,
                    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                    n_gait_cycles=-1, gait_style="auto",
                    trimming_start=trim_start, trimming_end=trim_end
                )
                scal = ga.compute_scalars(scalar_names, return_all=True)

                cad = _to_clean_array(scal["cadence"]["value"]) if "cadence" in scal else np.array([])
                cad = cad[cad > 0]

                raw_step_length = scal["step_length"]["value"] if "step_length" in scal else None
                step_length = _extract_step_length_array(raw_step_length)

                ds = _to_clean_array(scal["double_support_time"]["value"]) if "double_support_time" in scal else np.array([])

                trunk_ap_acc = extract_trunk_ap_acceleration_series(ga, filt_freq=10)

                if trunk_ap_acc.size: all_trunk_ap_acc.append(trunk_ap_acc)
                if cad.size: all_cad.append(cad)
                if step_length.size: all_step_length.append(step_length)
                if ds.size: all_ds.append(ds)

            processed.append(pair["name"])
            if verbose:
                n = sum(len(c) for c in all_cad)
                print(f"  ✓ {pair['name']}  — ciclos R+L: {n}")

        except Exception as e:
            errors.append({"name": pair["name"], "error": str(e)})
            if verbose:
                print(f"  ✗ {pair['name']}  — error: {e}")
        finally:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

    cadences_raw = np.concatenate(all_cad) if all_cad else np.array([])
    stride_times = 120.0 / cadences_raw if cadences_raw.size else np.array([])
    step_lengths = np.concatenate(all_step_length) if all_step_length else None
    double_support_times = np.concatenate(all_ds) if all_ds else None
    trunk_ap_acc = np.concatenate(all_trunk_ap_acc) if all_trunk_ap_acc else None

    if verbose:
        print(f"  Ensayos OK : {len(processed)}/{len(pairs)}")
        print(f"  Tiempos de zancada totales para entropía: {len(stride_times)}")

    return {
        "stride_times": stride_times,
        "step_lengths": step_lengths,
        "double_support_times": double_support_times,
        "cadences": cadences_raw if cadences_raw.size else None,
        "trunk_ap_accelerations": trunk_ap_acc,
        "n_trials": len(processed),
        "trial_names": processed,
        "errors": errors,
    }

# ═══════════════════════════════════════════════════════════
# 7. FUNCIÓN PRINCIPAL: carpeta de sesión
# ═══════════════════════════════════════════════════════════
def analyze_fall_risk_from_session_folder(session_folder: str,
                                          gait_analysis_fn,
                                          max_mse_scale: int = 5,
                                          verbose: bool = True,
                                          fall_history: bool = False,
                                          height: float = 0.0) -> dict:
    series = collect_series_from_session(
        session_folder,
        gait_analysis_fn,
        verbose=verbose
    )

    st = series["stride_times"]
    if len(st) == 0:
        raise ValueError("No se pudieron extraer stride_times de ningún ensayo.")

    stats = compute_gait_stats(
        st,
        step_lengths=series.get("step_lengths"),
        cadences=series.get("cadences"),
        double_support_times=series.get("double_support_times"),
    )

    entropies = compute_all_entropies(
        stride_times=st,
        trunk_ap_acc=series.get("trunk_ap_accelerations"),
        max_scale=max_mse_scale,
    )

    risk = fall_risk_score(
        stats,
        entropies,
        fall_history=fall_history,
        height=height,
    )

    result = {
        "gait_stats": stats,
        "entropies": entropies,
        "fall_risk": risk,
        "n_trials": series["n_trials"],
        "trial_names": series["trial_names"],
        "errors": series["errors"],
    }

    if verbose:
        _print_report(stats, entropies, risk, n=len(st), n_trials=series["n_trials"])

    return result

# ═══════════════════════════════════════════════════════════
# 8. FUNCIÓN DE COMPATIBILIDAD: ensayo único
# ═══════════════════════════════════════════════════════════
def analyze_fall_risk_from_gait_analysis(scal_r_all: dict,
                                         scal_l_all: dict,
                                         max_mse_scale: int = 5,
                                         verbose: bool = True,
                                         fall_history: bool = False,
                                         height: float = 0.0) -> dict:
    """
    Análisis desde scalars de un único ensayo sin carpeta de sesión.
    Usa solo:
    - cadence
    - step_length
    - double_support_time
    y deriva stride_times desde cadence.
    """

    cadr = _to_clean_array(scal_r_all["cadence"]["value"]) if "cadence" in scal_r_all else np.array([])
    cadl = _to_clean_array(scal_l_all["cadence"]["value"]) if "cadence" in scal_l_all else np.array([])
    cad = np.concatenate([cadr, cadl]) if (cadr.size or cadl.size) else np.array([])
    cad = cad[cad > 0]

    if cad.size == 0:
        raise ValueError("No se pudo obtener cadence.")

    stride_times = 120.0 / cad

    step_r = _to_clean_array(scal_r_all["step_length"]["value"]) if "step_length" in scal_r_all else np.array([])
    step_l = _to_clean_array(scal_l_all["step_length"]["value"]) if "step_length" in scal_l_all else np.array([])
    step = np.concatenate([step_r, step_l]) if (step_r.size or step_l.size) else None

    ds_r = _to_clean_array(scal_r_all["double_support_time"]["value"]) if "double_support_time" in scal_r_all else np.array([])
    ds_l = _to_clean_array(scal_l_all["double_support_time"]["value"]) if "double_support_time" in scal_l_all else np.array([])
    ds = np.concatenate([ds_r, ds_l]) if (ds_r.size or ds_l.size) else None

    stats = compute_gait_stats(
        stride_times,
        step_lengths=step,
        cadences=cad,
        double_support_times=ds,
    )

    entropies = compute_all_entropies(
        stride_times=stride_times,
        max_scale=max_mse_scale,
    )

    risk = fall_risk_score(
        stats,
        entropies,
        fall_history=fall_history,
        height=height,
    )

    result = {
        "gait_stats": stats,
        "entropies": entropies,
        "fall_risk": risk,
        "n_trials": 1,
    }

    if verbose:
        _print_report(stats, entropies, risk, n=len(stride_times), n_trials=1)

    return result

# ═══════════════════════════════════════════════════════════
# 9. REPORTE EN CONSOLA
# ═══════════════════════════════════════════════════════════
def _fmt(v, d=3):
    return f"{v:.{d}f}" if (v is not None and not np.isnan(float(v))) else "N/A"

def _print_report(stats, entropies, risk, n=0, n_trials=None):
    print("\nCLASIFICACIÓN CLÍNICA")
    print(f" Nivel de riesgo : {risk.get('risk_level', 'N/D')}")
    _sb = risk.get('score_base', '')
    _sf = risk.get('score_final', '')
    _fh = risk.get('fall_history_applied', False)

    if _sb != '' and _sf != '':
        hist_str = f" (+15 historial -> {_sf})" if _fh else ""
        print(f" Score : {_sb}/100{hist_str}")

    if n_trials:
        print(f" Ensayos usados : {n_trials} | Puntos: {n}")

    print("\nFACTORES CLÍNICOS CONTRIBUYENTES")
    for f in risk.get("contributing_factors", []):
        print(f" - {f}")

    note = risk.get("_entropy_note", "")
    if note:
        print(f"\nENTROPÍAS (DESCRIPTIVAS)\n {note}")

    print("\nINTERPRETACIÓN")
    print(" ", risk.get("interpretation", ""))
    print("=" * 60)