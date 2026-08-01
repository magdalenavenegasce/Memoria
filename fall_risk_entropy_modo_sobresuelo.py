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
def normalize_stride_times_by_speed(stride_times: np.ndarray,
                                    walking_speeds: Optional[np.ndarray]) -> tuple:
    """
    Normaliza la serie de tiempo de zancada por la velocidad media de marcha.

    Retorna
    -------
    stride_times_norm : np.ndarray
    speed_mean : float
    """
    st = _to_clean_array(stride_times)
    ws = _to_clean_array(walking_speeds) if walking_speeds is not None else np.array([])

    speed_mean = float(np.mean(ws)) if ws.size > 0 else np.nan

    if st.size == 0:
        return st, speed_mean

    if not np.isfinite(speed_mean) or speed_mean <= 0:
        return st.copy(), speed_mean

    return st / speed_mean, speed_mean

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

def normalize_trunk_ap_acceleration_by_speed(trunk_ap_acc: np.ndarray,
                                             walking_speeds: Optional[np.ndarray]) -> tuple:
    """
    Normaliza la aceleración AP del tronco/COM por la velocidad media de marcha.

    Retorna
    -------
    trunk_ap_acc_norm : np.ndarray
    speed_mean : float
    """
    acc = _to_clean_array(trunk_ap_acc)
    ws = _to_clean_array(walking_speeds) if walking_speeds is not None else np.array([])

    speed_mean = float(np.mean(ws)) if ws.size > 0 else np.nan

    if acc.size == 0:
        return acc, speed_mean

    if not np.isfinite(speed_mean) or speed_mean <= 0:
        return acc.copy(), speed_mean

    return acc / speed_mean, speed_mean

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
    MSE sobre max_scale escalas.
    Retorna {"by_scale": {1:val,...}, "auc_mse": float}
    Con pocos datos reduce max_scale para garantizar AUC válido.
    """
    x = np.array(time_series, dtype=float)
    N = len(x)
    effective_max = min(max_scale, max(1, N // 8))
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    vals = {}
    for s in range(1, effective_max + 1):
        cg = _coarse_grain(x, s)
        vals[s] = sample_entropy(cg, m=m, r=r) if len(cg) >= 5 else np.nan
    valid = [v for v in vals.values() if not np.isnan(v)]
    auc   = float(np.trapz(valid) / len(valid)) if len(valid) >= 2 else np.nan
    return {"by_scale": vals, "auc_mse": auc}

# ═══════════════════════════════════════════════════════════
# 3. ENTROPÍAS PARA TODAS LAS VARIABLES
# ═══════════════════════════════════════════════════════════
def compute_all_entropies(stride_times: np.ndarray,
                          walking_speeds: Optional[np.ndarray] = None,
                          trunk_ap_acc: Optional[np.ndarray] = None,
                          max_scale: int = 5) -> dict:
    out = {}

    out["stride_time"] = compute_stride_time_entropy(
        stride_times=stride_times,
        walking_speeds=walking_speeds,
        max_scale=max_scale,
        m=2,
        r=None
    )

    out["stride_time_variability"] = compute_stride_time_variability_entropy(
        stride_times=stride_times,
        walking_speeds=walking_speeds,
        window=5,
        m=2,
        r=None
    )

    out["trunk_ap_acceleration"] = compute_trunk_ap_mse(
        trunk_ap_acc=trunk_ap_acc,
        walking_speeds=walking_speeds,
        max_scale=max_scale,
        m=2,
        r=None
    )

    return out

# ═══════════════════════════════════════════════════════════
# 3A. ENTROPÍA ESPECÍFICA DE TIEMPO DE ZANCADA
# ═══════════════════════════════════════════════════════════
def compute_stride_time_entropy(stride_times: np.ndarray,
                                walking_speeds: Optional[np.ndarray] = None,
                                max_scale: int = 5,
                                m: int = 2,
                                r: Optional[float] = None) -> dict:
    """
    Calcula SampEn y MSE para la serie de tiempo de zancada,
    normalizada por la velocidad media de marcha.
    """
    st_raw = _to_clean_array(stride_times)
    st_norm, speed_mean = normalize_stride_times_by_speed(st_raw, walking_speeds)

    if st_norm.size < 10:
        return {
            "sampen": np.nan,
            "mse": {"by_scale": {s: np.nan for s in range(1, max_scale + 1)},
                    "auc_mse": np.nan},
            "n": int(st_norm.size),
            "mean_raw": float(np.mean(st_raw)) if st_raw.size else np.nan,
            "sd_raw": float(np.std(st_raw, ddof=1)) if st_raw.size >= 2 else np.nan,
            "cv_pct_raw": (float(np.std(st_raw, ddof=1)) / float(np.mean(st_raw)) * 100.0)
                          if st_raw.size >= 2 and np.mean(st_raw) > 0 else np.nan,
            "mean_norm": float(np.mean(st_norm)) if st_norm.size else np.nan,
            "sd_norm": float(np.std(st_norm, ddof=1)) if st_norm.size >= 2 else np.nan,
            "speed_mean": speed_mean,
            "normalized_by_speed": True,
            "r_used": np.nan
        }

    sd_norm = float(np.std(st_norm, ddof=1))
    mean_norm = float(np.mean(st_norm))
    r_used = float(0.2 * sd_norm) if r is None else float(r)

    return {
        "sampen": sample_entropy(st_norm, m=m, r=r_used),
        "mse": multiscale_entropy(st_norm, max_scale=max_scale, m=m, r=r_used),
        "n": int(st_norm.size),
        "mean_raw": float(np.mean(st_raw)),
        "sd_raw": float(np.std(st_raw, ddof=1)),
        "cv_pct_raw": (float(np.std(st_raw, ddof=1)) / float(np.mean(st_raw)) * 100.0)
                      if np.mean(st_raw) > 0 else np.nan,
        "mean_norm": mean_norm,
        "sd_norm": sd_norm,
        "speed_mean": speed_mean,
        "normalized_by_speed": True,
        "r_used": r_used
    }

# ═══════════════════════════════════════════════════════════
# 3B. ENTROPÍA ESPECÍFICA DE VARIABILIDAD DEL TIEMPO DE ZANCADA
# ═══════════════════════════════════════════════════════════
def build_stride_time_variability_series(stride_times: np.ndarray,
                                         walking_speeds: Optional[np.ndarray] = None,
                                         window: int = 5) -> dict:
    """
    Construye una serie de variabilidad local del tiempo de zancada
    usando CV rolling (%), y la normaliza por velocidad media de marcha.
    """
    st = _to_clean_array(stride_times)
    ws = _to_clean_array(walking_speeds) if walking_speeds is not None else np.array([])

    speed_mean = float(np.mean(ws)) if ws.size > 0 else np.nan

    if st.size < max(window + 2, 6):
        return {
            "raw_series": np.array([]),
            "norm_series": np.array([]),
            "speed_mean": speed_mean,
            "window": window,
            "n": 0,
            "normalized_by_speed": True
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

    if raw_series.size == 0:
        norm_series = np.array([])
    elif np.isfinite(speed_mean) and speed_mean > 0:
        norm_series = raw_series / speed_mean
    else:
        norm_series = raw_series.copy()

    return {
        "raw_series": raw_series,
        "norm_series": norm_series,
        "speed_mean": speed_mean,
        "window": window,
        "n": int(norm_series.size),
        "normalized_by_speed": True
    }

def compute_stride_time_variability_entropy(stride_times: np.ndarray,
                                            walking_speeds: Optional[np.ndarray] = None,
                                            window: int = 5,
                                            m: int = 2,
                                            r: Optional[float] = None) -> dict:
    """
    Calcula SampEn de la variabilidad local del tiempo de zancada
    (serie rolling de CV), normalizada por velocidad media.
    """
    var_data = build_stride_time_variability_series(
        stride_times=stride_times,
        walking_speeds=walking_speeds,
        window=window
    )

    raw_series = var_data["raw_series"]
    norm_series = var_data["norm_series"]
    speed_mean = var_data["speed_mean"]

    if norm_series.size < 10:
        return {
            "sampen": np.nan,
            "n": int(norm_series.size),
            "window": window,
            "series_mean_raw": float(np.mean(raw_series)) if raw_series.size else np.nan,
            "series_sd_raw": float(np.std(raw_series, ddof=1)) if raw_series.size >= 2 else np.nan,
            "series_mean_norm": float(np.mean(norm_series)) if norm_series.size else np.nan,
            "series_sd_norm": float(np.std(norm_series, ddof=1)) if norm_series.size >= 2 else np.nan,
            "speed_mean": speed_mean,
            "normalized_by_speed": True,
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
        "speed_mean": speed_mean,
        "normalized_by_speed": True,
        "r_used": r_used
    }

# ═══════════════════════════════════════════════════════════
# 3C. ENTROPÍA ESPECÍFICA DE ACELERACIÓN AP DEL TRONCO/COM
# ═══════════════════════════════════════════════════════════
def compute_trunk_ap_mse(trunk_ap_acc: np.ndarray,
                         walking_speeds: Optional[np.ndarray] = None,
                         max_scale: int = 5,
                         m: int = 2,
                         r: Optional[float] = None) -> dict:
    """
    Calcula MSE para aceleración AP del tronco/COM,
    normalizada por velocidad media de marcha.
    """
    acc_raw = _to_clean_array(trunk_ap_acc)
    acc_norm, speed_mean = normalize_trunk_ap_acceleration_by_speed(
        trunk_ap_acc=acc_raw,
        walking_speeds=walking_speeds
    )

    if acc_raw.size == 0:
        return {
            "mse": {
                "by_scale": {s: np.nan for s in range(1, max_scale + 1)},
                "auc_mse": np.nan
            },
            "n": 0,
            "mean_raw": np.nan,
            "sd_raw": np.nan,
            "mean_norm": np.nan,
            "sd_norm": np.nan,
            "speed_mean": speed_mean,
            "normalized_by_speed": True,
            "r_used": np.nan
        }

    if acc_norm.size < 10:
        return {
            "mse": {
                "by_scale": {s: np.nan for s in range(1, max_scale + 1)},
                "auc_mse": np.nan
            },
            "n": int(acc_norm.size),
            "mean_raw": float(np.mean(acc_raw)),
            "sd_raw": float(np.std(acc_raw, ddof=1)) if acc_raw.size >= 2 else np.nan,
            "mean_norm": float(np.mean(acc_norm)),
            "sd_norm": float(np.std(acc_norm, ddof=1)) if acc_norm.size >= 2 else np.nan,
            "speed_mean": speed_mean,
            "normalized_by_speed": True,
            "r_used": np.nan
        }

    sd_norm = float(np.std(acc_norm, ddof=1))
    r_used = float(0.2 * sd_norm) if r is None else float(r)

    return {
        "mse": multiscale_entropy(acc_norm, max_scale=max_scale, m=m, r=r_used),
        "n": int(acc_norm.size),
        "mean_raw": float(np.mean(acc_raw)),
        "sd_raw": float(np.std(acc_raw, ddof=1)) if acc_raw.size >= 2 else np.nan,
        "mean_norm": float(np.mean(acc_norm)),
        "sd_norm": sd_norm,
        "speed_mean": speed_mean,
        "normalized_by_speed": True,
        "r_used": r_used
    }

# ═══════════════════════════════════════════════════════════
# 4. ESTADÍSTICOS DE MARCHA
# ═══════════════════════════════════════════════════════════

def compute_gait_stats(stride_times: np.ndarray,
                       stride_lengths: Optional[np.ndarray],
                       walking_speeds: Optional[np.ndarray],
                       cadences: Optional[np.ndarray] = None) -> dict:
    out = {}
    out["stride_time_mean_s"]  = float(np.mean(stride_times))
    out["stride_time_sd_s"]    = float(np.std(stride_times, ddof=1))
    out["stride_time_cv_pct"]  = (out["stride_time_sd_s"] / out["stride_time_mean_s"] * 100
                                  if out["stride_time_mean_s"] > 0 else np.nan)
    if stride_lengths is not None and len(stride_lengths) > 0:
        out["stride_length_mean_m"] = float(np.mean(stride_lengths))
        out["stride_length_sd_m"]   = float(np.std(stride_lengths, ddof=1))
        out["stride_length_cv_pct"] = (out["stride_length_sd_m"] / out["stride_length_mean_m"] * 100
                                       if out["stride_length_mean_m"] > 0 else np.nan)
    if walking_speeds is not None and len(walking_speeds) > 0:
        out["walking_speed_mean_ms"] = float(np.mean(walking_speeds))
        out["walking_speed_sd_ms"]   = float(np.std(walking_speeds, ddof=1))
    if cadences is not None and len(cadences) > 0:
        out["cadence_mean_spm"] = float(np.mean(cadences))
        out["cadence_sd_spm"]   = float(np.std(cadences, ddof=1))
        out["cadence_cv_pct"]   = (out["cadence_sd_spm"] / out["cadence_mean_spm"] * 100
                                   if out["cadence_mean_spm"] > 0 else np.nan)
    return out

# ═══════════════════════════════════════════════════════════
# 5. SCORE NUMÉRICO ACUMULADO
# ═══════════════════════════════════════════════════════════

def _score_cv_stride_time(cv: float) -> tuple:
    """Bloque A — Hausdorff et al. 2001 (APMR). Máx 22 pts."""
    if np.isnan(cv): return 0, None
    if cv > 6.0:   return 22, f"CV Tiempo de zancada muy alto ({cv:.1f}%) [>6.0%]"
    elif cv > 4.0: return 12, f"CV Tiempo de zancada elevado ({cv:.1f}%) [>4.0%]"
    elif cv > 2.5: return  5, f"CV Tiempo de zancada limítrofe ({cv:.1f}%) [>2.5%]"
    return 0, None

def _score_walking_speed(ws: float) -> tuple:
    """Bloque B — Studenski et al. 2011 (JAMA); Cesari 2005; Bohannon 1997. Máx 22 pts."""
    if np.isnan(ws): return 0, None
    if ws < 0.6:   return 22, f"Velocidad de la Marcha muy reducida ({ws:.2f} m/s) [<0.60]"
    elif ws < 0.8: return 17, f"Velocidad reducida ({ws:.2f} m/s) [<0.80]"
    elif ws < 1.0: return 10, f"Velocidad limítrofe ({ws:.2f} m/s) [<1.00]"
    elif ws < 1.2: return  3, f"Velocidad normal-baja ({ws:.2f} m/s) [<1.20]"
    return 0, None

def _score_stride_length(sl: float, height: float) -> tuple:
    """Bloque C — Oberg et al. 1993; Bohannon 1997. Máx 20 pts."""
    if np.isnan(sl): return 0, None
    if height > 0:
        ratio = sl / height
        if ratio < 0.50:   return 20, f"Longitud de zancada muy corta ({sl:.2f} m, ratio={ratio:.2f}) [<0.50]"
        elif ratio < 0.60: return 10, f"Longitud de zancada corta ({sl:.2f} m, ratio={ratio:.2f}) [<0.60]"
        elif ratio < 0.72: return  3, f"Longitud de zancada normal-baja ({sl:.2f} m, ratio={ratio:.2f}) [<0.72]"
        return 0, None
    else:
        if sl < 0.9:   return 20, f"Longitud de zancada muy corta ({sl:.2f} m) [<0.90]"
        elif sl < 1.1: return 10, f"Longitud de zancada corta ({sl:.2f} m) [<1.10]"
        elif sl < 1.3: return  3, f"Longitud de zancada normal-baja ({sl:.2f} m) [<1.30]"
        return 0, None

def score_cadence(cad: float) -> tuple:
    if np.isnan(cad):
        return 0, None
    if cad < 90:
        return 15, f"Cadencia muy reducida ({cad:.1f} spm) [<90]"
    elif cad < 100:
        return 8, f"Cadencia reducida ({cad:.1f} spm) [90-100]"
    return 0, None

def _score_cv_cadence(cv_cad: float) -> tuple:
    """Bloque E — Lord et al. 2011. Máx 12 pts."""
    if np.isnan(cv_cad): return 0, None
    if cv_cad > 8.0:   return 12, f"CV Cadencia muy alto ({cv_cad:.1f}%) [>8.0%]"
    elif cv_cad > 5.0: return  7, f"CV Cadencia elevado ({cv_cad:.1f}%) [>5.0%]"
    return 0, None

def _score_cv_stride_length(cv_sl: float) -> tuple:
    """Bloque F — Hausdorff et al. 2001. Máx 9 pts."""
    if np.isnan(cv_sl): return 0, None
    if cv_sl > 6.0:   return 9, f"CV Longitud de zancada muy alto ({cv_sl:.1f}%) [>6.0%]"
    elif cv_sl > 4.0: return 5, f"CV Longitud de zancada elevado ({cv_sl:.1f}%) [>4.0%]"
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
        ("stride_time", 30, 15, 1.20, 1.50, 1.00, 1.70, True, "literature_synthesis"),
        ("stride_time_variability", 20, 10, 1.20, 2.00, 1.00, 2.50, False, "provisional"),
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
        ("stride_time", 30, 15, 0.92, 1.15, 0.82, 1.28, False, "provisional"),
        ("trunk_ap_acceleration", 20, 10, 0.88, 1.12, 0.78, 1.25, True, "semi_literature_based"),
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

def _combined_interpretation(score_base: float, risk_level: str,
                             score_sampen: float, score_aucmse: float,
                             entropies: dict) -> str:
    """
    Genera un párrafo de interpretación clínica combinando el score clínico
    con los scores de entropía (SampEn y AUC-MSE).
    """
    import math

    def _safe(v):
        try:
            return float(v) if v is not None and not math.isnan(float(v)) else None
        except:
            return None

    sb = _safe(score_base)
    sse = _safe(score_sampen)
    sauc = _safe(score_aucmse)
    lv = str(risk_level).upper()

    # Valores descriptivos de entropía por señal
    st_se = _safe(entropies.get("stride_time", {}).get("sampen"))
    stv_se = _safe(entropies.get("stride_time_variability", {}).get("sampen"))
    st_auc = _safe(entropies.get("stride_time", {}).get("mse", {}).get("auc_mse"))
    ta_auc = _safe(entropies.get("trunk_ap_acceleration", {}).get("mse", {}).get("auc_mse"))

    se_str = f"{sse:.0f}/50" if sse is not None else "N/D"
    auc_str = f"{sauc:.0f}/50" if sauc is not None else "N/D"

    # ── Determinar patrón de entropía ──────────────────────────────────────
    # "rígido" = SampEn bajo (automatización)
    # "caótico" = SampEn alto (inestabilidad)
    # "normal"  = dentro de rango
    patron = "normal"
    if sse is not None:
        if sse <= 20:
            patron = "rigido"
        elif sse >= 40:
            patron = "caotico"

    # ── Construir párrafo según combinación ───────────────────────────────
    if lv == "ALTO" and patron == "rigido":
        parrafo = (
            f"El score clínico indica riesgo ALTO de caída, respaldado por alta variabilidad "
            f"espacio-temporal en múltiples parámetros. La complejidad dinámica de la marcha "
            f"es baja (Score SampEn: {se_str}), patrón consistente con automatización "
            f"compensatoria: el control locomotor es rígido y predecible, lo que reduce la "
            f"capacidad de adaptación ante perturbaciones del entorno. Este perfil se asocia "
            f"a mayor riesgo prospectivo de caída (Hausdorff 2001, Lipsitz 2004)."
        )
    elif lv == "ALTO" and patron == "caotico":
        parrafo = (
            f"El score clínico indica riesgo ALTO de caída. La complejidad dinámica es "
            f"elevada (Score SampEn: {se_str}), lo que en combinación con alta variabilidad "
            f"clínica sugiere inestabilidad locomotora activa: la señal irregular refleja "
            f"pérdida del control neuromuscular fino. Este patrón —alta variabilidad con "
            f"alta entropía— es indicativo de marcha caótica y desorganizada "
            f"(Stergiou & Decker 2011, Hausdorff 2001)."
        )
    elif lv == "ALTO" and patron == "normal":
        parrafo = (
            f"El score clínico indica riesgo ALTO de caída. La complejidad dinámica de la "
            f"marcha está conservada (Score SampEn: {se_str}; Score AUC-MSE: {auc_str}), "
            f"lo que indica que el riesgo no está mediado por rigidez locomotora ni "
            f"inestabilidad dinámica, sino por alta variabilidad en los parámetros "
            f"espacio-temporales (zancada inconsistente, velocidad reducida o longitud "
            f"de paso alterada). Este patrón —variabilidad alta con complejidad conservada— "
            f"refleja un control locomotor activo pero ineficiente, asociado a riesgo de "
            f"tropiezos y caídas por inconsistencia del paso (Hausdorff 2001, Brach 2010)."
        )
    elif lv == "MODERADO" and patron == "rigido":
        parrafo = (
            f"El score clínico indica riesgo MODERADO. Sin embargo, la baja complejidad "
            f"dinámica (Score SampEn: {se_str}) sugiere un patrón de marcha automatizado "
            f"y rígido que puede no reflejarse completamente en los parámetros espacio-"
            f"temporales. Se recomienda seguimiento clínico y evaluación funcional periódica "
            f"(Lipsitz 2004, Lord 2011)."
        )
    elif lv == "MODERADO" and patron == "caotico":
        parrafo = (
            f"El score clínico indica riesgo MODERADO. La complejidad dinámica elevada "
            f"(Score SampEn: {se_str}) añade una señal de alerta adicional: la variabilidad "
            f"irregular en el patrón de zancada puede indicar inestabilidad incipiente no "
            f"capturada por los parámetros clínicos estándar. Se recomienda valoración "
            f"funcional (Stergiou & Decker 2011)."
        )
    elif lv == "MODERADO":
        parrafo = (
            f"El score clínico indica riesgo MODERADO, con algunos parámetros de marcha "
            f"alterados. La complejidad dinámica (Score SampEn: {se_str}; Score AUC-MSE: "
            f"{auc_str}) no añade señales de alarma adicionales. Se recomienda seguimiento "
            f"preventivo y reevaluación periódica (Kressig 2004, Lord 2011)."
        )
    elif lv == "BAJO":
        parrafo = (
            f"Los parámetros espacio-temporales se encuentran dentro de rangos normativos "
            f"(score clínico: {sb:.0f}/100). La complejidad dinámica de la marcha es "
            f"adecuada (Score SampEn: {se_str}; Score AUC-MSE: {auc_str}), sin señales de "
            f"rigidez locomotora ni inestabilidad dinámica. Se recomienda mantener "
            f"seguimiento preventivo anual (Studenski 2011, Cesari 2005)."
        )
    else:
        parrafo = (
            f"Score clínico: {sb:.0f}/100 ({lv}). Score SampEn: {se_str}. "
            f"Score AUC-MSE: {auc_str}. Interpretación combinada no disponible."
        )

    return parrafo


def fall_risk_score(gait_stats: dict, entropies: dict,
                    fall_history: bool = False,
                    height: float = 0.0) -> dict:
    """
    Score numérico acumulado 0–100 con ajuste por historial de caída.
    Cada bloque suma puntos según umbrales validados en literatura.
    Con historial: +15 puntos (cap 100), ajuste pragmático basado en que
    el antecedente de caída es un fuerte predictor de nuevas caídas (p.ej., Tinetti 1988).
    Clasificación: >= 60 ALTO, >= 30 MODERADO, < 30 BAJO.
    """
    score = 0.0
    factors = []

    cv = gait_stats.get("stride_time_cv_pct", np.nan)
    ws = gait_stats.get("walking_speed_mean_ms", np.nan)
    sl = gait_stats.get("stride_length_mean_m", np.nan)
    cv_c = gait_stats.get("cadence_cv_pct", np.nan)
    cv_sl = gait_stats.get("stride_length_cv_pct", np.nan)
    cad = gait_stats.get("cadence_mean_spm", np.nan)

    # Bloque A
    pts, msg = _score_cv_stride_time(cv)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque A - CV Tiempo de zancada] valor={cv:.2f} -> pts={pts}")

    # Bloque B
    pts, msg = _score_walking_speed(ws)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque B - Velocidad de caminata] valor={ws:.2f} -> pts={pts}")
    # Bloque C
    pts, msg = _score_stride_length(sl, height)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque C - Longitud de zancada] valor={sl:.2f} -> pts={pts}")

    # Bloque D
    pts, msg = score_cadence(cad)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque D - Cadencia] valor={cad:.2f} -> pts={pts}")

    # Bloque E
    pts, msg = _score_cv_cadence(cv_c)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque E - CV Cadencia] valor={cv_c:.2f} -> pts={pts}")

    # Bloque F
    pts, msg = _score_cv_stride_length(cv_sl)
    score += pts
    if msg:
        factors.append(msg)
        print(f"[Bloque F - CV Longitud de zancada] valor={cv_sl:.2f} -> pts={pts}")

    # Score Entropía independiente (SampEn 0-50, AUC-MSE 0-50, Total 0-100)
    _se_result = _score_sampen_independent(entropies)
    _auc_result = _score_aucmse_independent(entropies)

    score_sampen = _se_result["score"]
    score_aucmse = _auc_result["score"]
    score_entropy_total = score_sampen + score_aucmse

    if score_entropy_total >= 60:
        entropy_level = "ALTO"
    elif score_entropy_total >= 30:
        entropy_level = "MODERADO"
    else:
        entropy_level = "BAJO"

    # Ajuste por historial de caída: +15 puntos (cap 100)
    score_base = round(score, 1)
    if fall_history:
        score = min(score + 15.0, 100.0)

    score = round(score, 1)

    # Clasificación
    if score >= 60:
        level = "ALTO"
    elif score >= 30:
        level = "MODERADO"
    else:
        level = "BAJO"

    # Notas de entropía para display
    entropy_note_parts = []

    for _bd in _se_result["breakdown"]:
        _sig_label = _bd["signal"].replace("_", " ").title()
        if not np.isnan(_bd["sampen"]):
            entropy_note_parts.append(
                f"SampEn {_sig_label}={_bd['sampen']:.3f} ({_bd['categoria']}, +{_bd['pts']} pts)"
            )

    for _bd in _auc_result["breakdown"]:
        _sig_label = _bd["signal"].replace("_", " ").title()
        if not np.isnan(_bd["auc_mse"]):
            entropy_note_parts.append(
                f"AUC-MSE {_sig_label}={_bd['auc_mse']:.3f} ({_bd['categoria']}, +{_bd['pts']} pts)"
            )

    return {
        "score_base": score_base,
        "score_final": score,
        "score_sampen": score_sampen,
        "score_aucmse": score_aucmse,
        "score_entropy_total": score_entropy_total,
        "entropy_level": entropy_level,
        "entropy_sampen_breakdown": _se_result["breakdown"],
        "entropy_aucmse_breakdown": _auc_result["breakdown"],
        "fall_history_applied": fall_history,
        "risk_level": level,
        "global_risk_level": level,
        "contributing_factors": factors if factors else ["Sin factores clínicos principales detectados"],
        "entropy_note": " | ".join(entropy_note_parts) if entropy_note_parts else "Sin datos de entropía",
        "interpretation": (
            "Score acumulado basado en umbrales validados en literatura "
            "(Hausdorff 2001, Studenski 2011, Kressig 2004, Tinetti 1988, Stergiou 2011). "
            + (
                f"Ajuste por historial de caída aplicado (+15 pts, score base={score_base})."
                if fall_history else "Sin historial de caída."
            )
        ),
        "combined_interpretation": _combined_interpretation(
            score_base, level, score_sampen, score_aucmse, entropies
        ),
    }

# ═══════════════════════════════════════════════════════════
# 6. RECOLECTAR SERIES DE TODOS LOS ENSAYOS
# ═══════════════════════════════════════════════════════════
def collect_series_from_session(session_folder: str,
                                gait_analysis_fn,
                                filter_frequency: int = 6,
                                trim_start: float = 0.05,
                                trim_end:   float = 0.05,
                                verbose: bool = True) -> dict:
    scalar_names = {
        'gait_speed', 'stride_length', 'cadence',
        'step_width', 'single_support_time',
        'double_support_time', 'step_length_symmetry'
    }
    pairs = _find_trial_pairs(session_folder)
    if not pairs:
        raise FileNotFoundError(f"No se encontraron pares TRC+MOT en: {session_folder}")

    all_cad, all_sl, all_sp, all_trunk_ap_acc = [], [], [], []
    processed, errors = [], []

    for pair in pairs:
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="OpenCapOffline_ent_"))
        try:
            (tmp_dir / "MarkerData").mkdir(parents=True)
            (tmp_dir / "OpenSimData" / "Kinematics").mkdir(parents=True)
            shutil.copy2(pair["trc"], str(tmp_dir / "MarkerData" / f"{pair['name']}.trc"))
            shutil.copy2(pair["mot"], str(tmp_dir / "OpenSimData" / "Kinematics" / f"{pair['name']}.mot"))

            for leg in ('r', 'l'):
                ga   = gait_analysis_fn(
                    str(tmp_dir), pair["name"], leg=leg,
                    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                    n_gait_cycles=-1, gait_style='auto',
                    trimming_start=trim_start, trimming_end=trim_end
                )
                scal = ga.compute_scalars(scalar_names, return_all=True)
                cad  = _to_clean_array(scal["cadence"]["value"])
                cad  = cad[cad > 0]
                sl   = _to_clean_array(scal["stride_length"]["value"])
                sp   = _to_clean_array(scal["gait_speed"]["value"])
                trunk_ap_acc = extract_trunk_ap_acceleration_series(ga, filt_freq=10)
                if trunk_ap_acc.size:
                    all_trunk_ap_acc.append(trunk_ap_acc)
                if cad.size: all_cad.append(cad)
                if sl.size:  all_sl.append(sl)
                if sp.size:  all_sp.append(sp)

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

    cadences_raw   = np.concatenate(all_cad) if all_cad else np.array([])
    stride_times   = 120.0 / cadences_raw    if cadences_raw.size else np.array([])
    stride_lengths = np.concatenate(all_sl)  if all_sl else None
    walking_speeds = np.concatenate(all_sp)  if all_sp else None
    trunk_ap_acc = np.concatenate(all_trunk_ap_acc) if all_trunk_ap_acc else None

    if verbose:
        print(f"  Ensayos OK : {len(processed)}/{len(pairs)}")
        print(f"  Tiempos de zancada totales para entropía: {len(stride_times)}")

    return {
        "stride_times":   stride_times,
        "stride_lengths": stride_lengths,
        "walking_speeds": walking_speeds,
        "cadences":       cadences_raw if cadences_raw.size else None,
        "trunk_ap_accelerations": trunk_ap_acc,
        "n_trials":       len(processed),
        "trial_names":    processed,
        "errors":         errors,
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
    """
    Análisis completo desde carpeta de sesión con todos los ensayos.

    Uso:
        result = analyze_fall_risk_from_session_folder(
            session_folder, gait_analysis,
            fall_history=True, height=1.72, verbose=True
        )
    """
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  Recolectando ensayos de: {session_folder}")
        print(f"{'─'*55}")

    series = collect_series_from_session(session_folder, gait_analysis_fn, verbose=verbose)

    st  = series["stride_times"]
    sl  = series["stride_lengths"]
    ws  = series["walking_speeds"]
    cad = series.get("cadences")
    ta = series.get("trunk_ap_accelerations")

    if len(st) == 0:
        raise ValueError("No se pudieron extraer stride_times de ningún ensayo.")

    stats     = compute_gait_stats(st, sl, ws, cad)
    entropies = compute_all_entropies(
        stride_times=st,
        walking_speeds=ws,
        trunk_ap_acc=ta,
        max_scale=max_mse_scale
    )
    risk      = fall_risk_score(stats, entropies, fall_history=fall_history, height=height)

    result = {
        "gait_stats":  stats,
        "entropies":   entropies,
        "fall_risk":   risk,
        "n_trials":    series["n_trials"],
        "ntrials":     series["n_trials"],
        "trial_names": series["trial_names"],
        "errors":      series["errors"],
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
    """Análisis desde scalars de un único ensayo (sin carpeta de sesión)."""
    cad = np.concatenate([
        _to_clean_array(scal_r_all["cadence"]["value"]),
        _to_clean_array(scal_l_all["cadence"]["value"])
    ])
    cad = cad[cad > 0]
    if cad.size == 0:
        raise ValueError("No se pudo obtener 'cadence'.")

    st   = 120.0 / cad
    sl_r = _to_clean_array(scal_r_all["stride_length"]["value"])
    sl_l = _to_clean_array(scal_l_all["stride_length"]["value"])
    sl   = np.concatenate([sl_r, sl_l]) if (sl_r.size + sl_l.size) > 0 else None
    sp_r = _to_clean_array(scal_r_all["gait_speed"]["value"])
    sp_l = _to_clean_array(scal_l_all["gait_speed"]["value"])
    ws   = np.concatenate([sp_r, sp_l]) if (sp_r.size + sp_l.size) > 0 else None
    
    stats     = compute_gait_stats(st, sl, ws, cad)
    entropies = compute_all_entropies(
        stride_times=st,
        walking_speeds=ws,
        max_scale=max_mse_scale
    )
    risk      = fall_risk_score(stats, entropies, fall_history=fall_history, height=height)

    result = {"gait_stats": stats, "entropies": entropies,
              "fall_risk": risk, "n_trials": 1, "ntrials": 1}
    if verbose:
        _print_report(stats, entropies, risk, n=len(st))
    return result

# ═══════════════════════════════════════════════════════════
# 9. REPORTE EN CONSOLA
# ═══════════════════════════════════════════════════════════
def _fmt(v, d=3):
    return f"{v:.{d}f}" if (v is not None and not np.isnan(float(v))) else "N/A"

def _print_report(stats, entropies, risk, n=0, n_trials=None):
    print("\nCLASIFICACIÓN CLÍNICA")
    print(f"  Nivel de riesgo : {risk.get('risk_level', 'N/D')}")
    sb = risk.get('score_base', '')
    sf = risk.get('score_final', '')
    fh = risk.get('fall_history_applied', False)
    if sb != '' and sf != '':
        hist_str = f"  (+ 15 historial → {sf})" if fh else ""
        print(f"  Score           : {sb}/100{hist_str}")
    if n_trials:
        print(f"  Ensayos usados  : {n_trials}  |  Puntos: {n}")
    print("\nFACTORES CLÍNICOS CONTRIBUYENTES")
    for f in risk.get("contributing_factors", []):
        print(f"  - {f}")
    note = risk.get("entropy_note", "")
    if note:
        print(f"\nENTROPÍAS (DESCRIPTIVAS)\n  {note}")
    print("\nINTERPRETACIÓN")
    print(" ", risk.get("interpretation", ""))
    print("=" * 60)