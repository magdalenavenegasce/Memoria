# -*- coding: utf-8 -*-
r"""
Análisis de Marcha — OFFLINE (sin OpenSim / sin API) con colorbars (plot_utils).
- Pide TRC y MOT locales + edad, altura, género.
- Prepara carpeta de sesión temporal con la estructura esperada:
  \MarkerData\<nombre>.trc
  \OpenSimData\Kinematics\<nombre>.mot
- Ejecuta gait_analysissinopensim.gait_analysis para R/L.
- Calcula PDKV y Foot Clearance desde TRC.
- Grafica resultados con color_bar_* (plot_utils) en la pestaña Resultados.
- Logo arriba a la izquierda.
- Exporta reporte PDF con portada (datos + logo), figura y texto descriptivo.
- NUEVO: Calcula Riesgo de Caída por Entropía usando carpeta de sesión real.
"""

import os, sys, re, shutil, tempfile, pathlib, math
import numpy as np
import pandas as pd
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
import textwrap as _tw, math as _math
from Gráficos_adicionales_modo_sobresuelo2 import generate_additional_report

# --- Logo (Pillow opcional) ---
try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

LOGO_PATH      = r"C:\opencap\opencap-processing\Logos\Logo MEDS 2023.png"
LOGO_MAX_WIDTH = 160

# --- Módulo de entropía (debe estar en el mismo directorio) ---
try:
    from fall_risk_entropy_modo_sobresuelo import (
        analyze_fall_risk_from_session_folder,
        analyze_fall_risk_from_gait_analysis,
    )
    _HAS_ENTROPY = True
except ImportError:
    _HAS_ENTROPY = False

# --- Clase gait_analysis sin OpenSim ---
try:
    from gait_analysissinopensim11 import gait_analysis as _gait_analysis_impl
except (ImportError, AttributeError):
    try:
        from gait_analysissinopensim10 import gait_analysis as _gait_analysis_impl
    except (ImportError, AttributeError) as err:
        try:
            from utilssinopensim import gait_analysis as _gait_analysis_impl
        except (ImportError, AttributeError):
            raise ImportError(
                "No se pudo importar 'gait_analysis'. Asegúrate de tener disponible "
                "gait_analysissinopensim5.py en el mismo directorio."
            ) from err

gait_analysis = _gait_analysis_impl

# --- Utilidades de color ---
from plot_utils_final2 import (
    color_bar_stride,
    color_bar_stepwidth,
    color_bar_cadence,
    #color_bar_symmetry,
    #color_bar_double_support,
    color_bar_speed,
    color_bar_pdkv,
    color_bar_footclearance,
)

# =========================
# Tablas de referencia
# =========================
reference_data = {
    "Hombre": {
        # Cadencia: Bohannon 1997 + Studenski 2011 para 81+
        "Cadence [steps/min]": {(18,30):(105.22,7.77),(31,40):(109.76,7.86),(41,50):(109.26,8.91),
                                (51,60):(107.08,9.19),(61,70):(110.99,4.16),(71,80):(111.08,9.99),
                                (81,120):(100.00,12.00)},
        # Stride length: Hollman 2011 + extrapolación para 81+
        "Step length [m]":    {(18,30):(0.75,0.14),(31,40):(0.80,0.15),(41,50):(0.79,0.12),
                                (51,60):(0.75,0.17),(61,70):(0.80,0.15),(71,80):(0.71,0.17),
                                (81,120):(0.62,0.16)},
        # Gait speed: Studenski 2011 (JAMA), corte clínico >1.12 m/s para adultos mayores
        "Gait speed [m/s]":   {(18,30):(1.30,0.26),(31,40):(1.42,0.26),(41,50):(1.41,0.21),
                                (51,60):(1.34,0.32),(61,70):(1.47,0.27),(71,80):(1.32,0.28),
                                (81,120):(1.10,0.24)},
    },
    "Mujer": {
        "Cadence [steps/min]": {(18,30):(111.88,9.25),(31,40):(114.41,9.45),(41,50):(115.14,8.75),
                                 (51,60):(114.35,8.36),(61,70):(118.87,8.21),(71,80):(111.03,17.04),
                                 (81,120):(100.00,14.00)},
        "Step length [m]":     {(18,30):(0.71,0.15),(31,40):(0.64,0.15),(41,50):(0.63,0.11),
                                 (51,60):(0.66,0.16),(61,70):(0.58,0.09),(71,80):(0.48,0.14),
                                 (81,120):(0.44,0.14)},
        "Gait speed [m/s]":    {(18,30):(1.32,0.25),(31,40):(1.24,0.26),(41,50):(1.28,0.25),
                                 (51,60):(1.29,0.29),(61,70):(1.23,0.20),(71,80):(0.99,0.27),
                                 (81,120):(0.88,0.22)},
    },
}

def get_reference(sex, metric, age):
    for (low, high), stats in reference_data[sex][metric].items():
        if low <= age <= high:
            return stats
    # Fuera de rango: usar el tramo más cercano (permite edades >80)
    best_key = min(reference_data[sex][metric].keys(),
                   key=lambda r: min(abs(age - r[0]), abs(age - r[1])))
    return reference_data[sex][metric][best_key]

# =========================================================
# === .TRC helpers (PDKV + Foot Clearance offline) ===
# =========================================================
def _split_ws(line: str):
    return re.findall(r"[^\s]+", line.strip())

def load_trc_opensim(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    hdr_idx = next((i for i, ln in enumerate(lines) if "Frame#" in ln), None)
    if hdr_idx is None:
        raise RuntimeError("No se encontró 'Frame#' en el encabezado del .trc.")
    header_row1 = _split_ws(lines[hdr_idx])
    header_row2 = _split_ws(lines[hdr_idx + 1])
    colnames = ["Frame#", "Time"]
    tokens1 = header_row1[2:]
    tokens2 = header_row2[2:] if len(header_row2) > 0 else []

    def norm_xyz(t):
        t = t.upper()
        return "X" if t.startswith("X") else "Y" if t.startswith("Y") else "Z" if t.startswith("Z") else t

    if tokens2 and len(tokens2) % 3 == 0:
        markers_expanded = tokens1 if len(tokens1) == len(tokens2) else sum(([m]*3 for m in tokens1), [])
        for m, ax in zip(markers_expanded, tokens2):
            colnames.append(f"{m}_{norm_xyz(ax)}")
        df = pd.read_csv(path, engine="python", sep=r"[\t ]+", header=None,
                         skiprows=hdr_idx+2, names=colnames)
    else:
        df = pd.read_csv(path, engine="python", sep=r"[\t ]+", header=None, skiprows=hdr_idx+2)
        ncols = df.shape[1]
        if (ncols - 2) % 3 != 0:
            raise RuntimeError("Formato TRC no reconocido (columnas != 2 + 3*n).")
        n_markers = (ncols - 2) // 3
        names = tokens1[:n_markers] if len(tokens1) >= n_markers else [f"M{i}" for i in range(1, n_markers+1)]
        new_cols = ["Frame#", "Time"]
        for nm in names:
            new_cols += [f"{nm}_X", f"{nm}_Y", f"{nm}_Z"]
        df.columns = new_cols
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def get_xyz(df: pd.DataFrame, base: str):
    return df[[f"{base}_X", f"{base}_Y", f"{base}_Z"]].to_numpy()

def compute_signed_dynamic_knee_valgus(H, K, A, T):
    eps = 1e-12
    HK = K - H; v1 = A - H; v2 = T - H
    normal = np.cross(v1, v2)
    n_norm = np.linalg.norm(normal, axis=1); n_norm[n_norm < eps] = eps
    normal = normal / n_norm[:, None]
    proj = HK - (np.sum(HK * normal, axis=1)[:, None] * normal)
    num = np.sum(HK * proj, axis=1)
    den = np.linalg.norm(HK, axis=1) * np.linalg.norm(proj, axis=1)
    den[den < eps] = eps
    cosang = np.clip(num / den, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    cross_vec = np.cross(proj, HK)
    sgn = np.sign(np.sum(normal * cross_vec, axis=1))
    return ang * sgn

def pdkv_from_markers(df: pd.DataFrame, hip, knee, ankle, toe, phase=(10, 30)):
    H = get_xyz(df, hip); K = get_xyz(df, knee)
    A = get_xyz(df, ankle); T = get_xyz(df, toe)
    signed_ang = compute_signed_dynamic_knee_valgus(H, K, A, T)
    n = len(signed_ang)
    i0 = max(0, min(int(phase[0]/100 * n), n-1))
    i1 = max(i0+1, min(int(phase[1]/100 * n), n))
    window = signed_ang[i0:i1]
    return float(np.nanmax(window)) if window.size else float("nan")

VERTICAL_AXIS = 'Y'

def _axis_col(base: str, axis: str) -> str:
    return f"{base}_{axis.upper()}"

def _median_dt(t: np.ndarray) -> float:
    dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else np.nan

# --- LPF helper ---
try:
    from scipy.signal import butter, filtfilt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _lowpass_1d(x: np.ndarray, fs: float, cutoff_hz: float = 8.0, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(fs) or fs <= 0:
        return x
    if _HAS_SCIPY:
        nyq = 0.5 * fs
        wn = min(0.99, max(1e-4, cutoff_hz / nyq))
        b, a = butter(order, wn, btype='low')
        try:
            return filtfilt(b, a, x, method='gust')
        except TypeError:
            return filtfilt(b, a, x)
    win = max(3, int(round(0.06 * fs)))
    if win % 2 == 0: win += 1
    k = np.ones(win, dtype=float) / win
    return np.convolve(x, k, mode='same')

def compute_foot_clearance_from_trc(df, toe_marker, heel_marker,
                                     vertical_axis=VERTICAL_AXIS,
                                     swing_threshold_mm=5.0,
                                     contact_min_separation_s=0.30,
                                     cutoff_hz=8.0):
    t = df['Time'].to_numpy()
    dt = _median_dt(t)
    if not np.isfinite(dt):
        raise RuntimeError("No se pudo estimar dt desde Time del TRC.")
    fs = 1.0 / dt
    toe_y  = _lowpass_1d(df[_axis_col(toe_marker,  vertical_axis)].to_numpy(), fs, cutoff_hz)
    heel_y = _lowpass_1d(df[_axis_col(heel_marker, vertical_axis)].to_numpy(), fs, cutoff_hz)
    rng = np.percentile(toe_y, 95) - np.percentile(toe_y, 5)
    s = 1000.0 if rng <= 3.0 else 1.0
    toe_mm, heel_mm = toe_y * s, heel_y * s
    foot_min = np.minimum(toe_mm, heel_mm)
    min_sep = max(1, int(round(contact_min_separation_s * fs)))
    cand = np.where((foot_min[1:-1] < foot_min[:-2]) & (foot_min[1:-1] <= foot_min[2:]))[0] + 1
    contacts = []
    for c in cand:
        if not contacts or c - contacts[-1] >= min_sep:
            contacts.append(int(c))
        elif foot_min[c] < foot_min[contacts[-1]]:
            contacts[-1] = int(c)
    swings = []
    n = len(foot_min)
    win = max(1, int(round(0.04 * fs)))
    for i in range(len(contacts) - 1):
        a, b = contacts[i], contacts[i+1]
        if b <= a + 2: continue
        a0, a1 = max(0, a-win), min(n, a+win+1)
        b0, b1 = max(0, b-win), min(n, b+win+1)
        ground_local = float(np.median(np.concatenate([foot_min[a0:a1], foot_min[b0:b1]])))
        toe_rel  = toe_mm  - ground_local
        heel_rel = heel_mm - ground_local
        foot_rel = np.minimum(toe_rel, heel_rel)
        pad = max(1, int(round(0.05 * (b - a))))
        lo, hi = a + pad, b - pad
        if hi <= lo: lo, hi = a, b
        mask = foot_rel[lo:hi+1] > swing_threshold_mm
        if not np.any(mask): continue
        toe_sub   = toe_rel[lo:hi+1][mask]
        fc_mm     = float(np.min(toe_sub))
        idx_local = int(np.argmin(toe_sub))
        idx_global = lo + np.where(mask)[0][idx_local]
        swings.append({"hs_i": int(a), "hs_j": int(b), "fc_idx": idx_global, "fc_mm": fc_mm})
    vals = [s["fc_mm"] for s in swings]
    return {"clearance_mm": vals, "events": swings,
            "summary": {"n_swings": len(vals),
                        "mean_mm":   float(np.mean(vals))   if vals else np.nan,
                        "median_mm": float(np.median(vals)) if vals else np.nan}}

def pdkv_by_stride_from_trc(df, side, phase_window=(50, 60), vertical_axis=VERTICAL_AXIS):
    side = side.upper()
    HIP, KNEE, ANK = f"{side}Hip", f"{side}Knee", f"{side}Ankle"
    TOE, HEEL = f"{side}BigToe", f"{side}Heel"
    H = get_xyz(df, HIP); K = get_xyz(df, KNEE)
    A = get_xyz(df, ANK); T = get_xyz(df, TOE)
    signed_ang = compute_signed_dynamic_knee_valgus(H, K, A, T)
    fc = compute_foot_clearance_from_trc(df, toe_marker=TOE, heel_marker=HEEL, vertical_axis=vertical_axis)
    hs_idx = sorted({e["hs_i"] for e in fc["events"]})
    peaks_deg, per_cycle = [], []
    for i in range(len(hs_idx) - 1):
        a, b = hs_idx[i], hs_idx[i+1]
        if b <= a + 2: continue
        i0 = max(a, min(int(a + (phase_window[0]/100.0)*(b-a)), b-1))
        i1 = max(i0+1, min(int(a + (phase_window[1]/100.0)*(b-a)), b))
        seg = signed_ang[i0:i1]
        if seg.size == 0 or not np.any(np.isfinite(seg)): continue
        peak = float(np.nanmax(seg))
        peaks_deg.append(peak)
        per_cycle.append({"hs_i": int(a), "hs_j": int(b), "idx0": int(i0), "idx1": int(i1), "peak_deg": peak})
    return {"peaks_deg": peaks_deg, "per_cycle": per_cycle,
            "summary": {"n_cycles":  len(peaks_deg),
                        "mean_deg":   float(np.mean(peaks_deg))   if peaks_deg else np.nan,
                        "median_deg": float(np.median(peaks_deg)) if peaks_deg else np.nan}}

# ======== UTILIDADES DE LOGO ========
from typing import Optional
SUPPORTED_RASTER_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def load_logo_as_array(path, max_width_px=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Logo no encontrado: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_RASTER_EXTS:
        raise ValueError(f"Formato no soportado: {ext}")
    if _HAS_PIL:
        img = Image.open(path).convert("RGBA")
        if max_width_px and img.width > max_width_px:
            scale = max_width_px / float(img.width)
            img = img.resize((max_width_px, int(round(img.height * scale))), Image.LANCZOS)
        return np.asarray(img, dtype=np.uint8)
    import matplotlib.image as mpimg
    arr = mpimg.imread(path)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr

def load_logo_image(path, max_width=160):
    if not _HAS_PIL or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        if img.width > max_width:
            scale = max_width / float(img.width)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None

# =========================================================
# Helpers offline
# =========================================================
def ensure_session_structure_from_files(trc_path: str, mot_path: str):
    trc_p = pathlib.Path(trc_path); mot_p = pathlib.Path(mot_path)
    if not trc_p.is_file(): raise FileNotFoundError(f"TRC no existe: {trc_p}")
    if not mot_p.is_file(): raise FileNotFoundError(f"MOT no existe: {mot_p}")
    trial_name  = trc_p.stem
    session_dir = pathlib.Path(tempfile.mkdtemp(prefix="OpenCapOffline_"))
    (session_dir / "MarkerData").mkdir(parents=True, exist_ok=True)
    (session_dir / "OpenSimData" / "Kinematics").mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(trc_p), str(session_dir / "MarkerData" / f"{trial_name}.trc"))
    shutil.copy2(str(mot_p), str(session_dir / "OpenSimData" / "Kinematics" / f"{trial_name}.mot"))
    return str(session_dir), trial_name

# =========================================================
# Pipeline principal
# =========================================================
def run_full_analysis_offline_colors(trc_path, mot_path, edad, sexo, altura_m, modo="Sobre suelo"):
    session_dir, trial_name = ensure_session_structure_from_files(trc_path, mot_path)

    scalar_names = {
        'gait_speed', 'stride_length', 'step_width', 'cadence',
        'single_support_time', 'double_support_time', 'step_length_symmetry',
    }
    filter_frequency = 6
    trim_start = 0.05
    trim_end   = 0.05

    ga_r = gait_analysis(session_dir, trial_name, leg='r',
                         lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                         n_gait_cycles=-1, gait_style='auto',
                         trimming_start=trim_start, trimming_end=trim_end)
    ga_l = gait_analysis(session_dir, trial_name, leg='l',
                         lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
                         n_gait_cycles=-1, gait_style='auto',
                         trimming_start=trim_start, trimming_end=trim_end)

    def arr_first(scal_dict, name):
        v = scal_dict[name]["value"]
        a = np.asarray(v, dtype=float).ravel()
        a = a[np.isfinite(a)]
        return float(a[0]) if a.size > 0 else np.nan
    
    def arr_last(scal_dict, name):
        v = scal_dict[name]['value']
        a = np.asarray(v, dtype=float).ravel()
        a = a[np.isfinite(a)]
        return float(a[-1]) if a.size > 0 else np.nan

    scal_r_all = ga_r.compute_scalars(scalar_names, return_all=True)
    scal_l_all = ga_l.compute_scalars(scalar_names, return_all=True)

    cadence_r_1  = arr_first(scal_r_all, "cadence")
    speed_r_1    = arr_first(scal_r_all, "gait_speed")
    stride_r_1   = arr_first(scal_r_all, "stride_length")
    stride_l_1   = arr_first(scal_l_all, "stride_length")
    stepw_r_1    = arr_first(scal_r_all, "step_width")

    df_trc = load_trc_opensim(trc_path)

    PDKV_R_info = pdkv_by_stride_from_trc(df_trc, 'R', phase_window=(50, 60))
    PDKV_L_info = pdkv_by_stride_from_trc(df_trc, 'L', phase_window=(50, 60))

    def _safe_first(arr_raw):
        a = np.asarray(arr_raw, dtype=float).ravel()
        a = a[np.isfinite(a)]
        return float(a[0]) if a.size > 0 else np.nan

    pdkv_r_1 = _safe_first(PDKV_R_info["peaks_deg"])
    pdkv_l_1 = _safe_first(PDKV_L_info["peaks_deg"])

    FC_R = compute_foot_clearance_from_trc(df_trc, "RBigToe", "RHeel")
    FC_L = compute_foot_clearance_from_trc(df_trc, "LBigToe", "LHeel")
    fc_r_1 = _safe_first(FC_R["clearance_mm"])
    fc_l_1 = _safe_first(FC_L["clearance_mm"])

    curves_r = ga_r.get_coordinates_normalized_time()
    curves_l = ga_l.get_coordinates_normalized_time()

    metrics = {
        "Cadencia [pasos/min]":           cadence_r_1,
        "Longitud de zancada [m]":             {"R": stride_r_1, "L": stride_l_1},
        "Ancho del paso [cm]":               stepw_r_1 * 100.0,
        "Velocidad Marcha [m/s]":              speed_r_1,
        "Valgo dinámico [deg]":{"R": pdkv_r_1,  "L": pdkv_l_1},
        "Amplitud del pie [mm]":           {"R": fc_r_1,    "L": fc_l_1},
    }

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 4.2*nrows), constrained_layout=True)
    axs = np.atleast_1d(axs).flatten()

    for i, (metric, val) in enumerate(metrics.items()):
        if "zancada" in metric:
            color_bar_stride(axs[i], val.get("R", np.nan), val.get("L", np.nan), float(altura_m), metric)
        elif "Ancho" in metric:
            color_bar_stepwidth(axs[i], val, label=metric, altura_paciente=altura_m)
        elif "Cadencia" in metric:
            color_bar_cadence(axs[i], val, metric)
        elif "Marcha" in metric:
            mean, sd = get_reference(sexo, "Gait speed [m/s]", int(edad))
            color_bar_speed(axs[i], val, mean, sd, metric)
        elif "Valgo" in metric:
            color_bar_pdkv(axs[i], val.get("R", np.nan), val.get("L", np.nan), metric)
        elif "Amplitud" in metric:
            color_bar_footclearance(axs[i], val.get("R", np.nan), val.get("L", np.nan), metric, int(edad))

    for j in range(len(metrics), len(axs)):
        axs[j].axis("off")

    handles = [plt.Line2D([0],[0], color=c, lw=2, label=l)
               for c, l in [("blue","Pierna Derecha"),("dimgrey","Pierna Izquierda"),("black","Valor Único")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)

    for ax in axs[:len(metrics)]:
        for side in ("top","right","left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_position(("data", -0.05))
        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax.grid(False)

    return fig, {
        "session_dir": session_dir,
        "trial_name":  trial_name,
        "metrics":     metrics,
        "curves_r":    curves_r,
        "curves_l":    curves_l,
        "scal_r_all":  scal_r_all,
        "scal_l_all":  scal_l_all,
    }

# =========================================================
# === FUNCIONES DE PLOTEO DE CINEMÁTICA ===
# =========================================================
def _find_kinematic_col(df, base, side=None, extras=None):
    cand = []
    if side in ("r", "l"):
        suf = f"_{side}"
        cand += [f"{base}{suf}", base]
        if extras: cand.extend([f"{e}{suf}" for e in extras])
    else:
        cand.append(base)
        if extras: cand.extend(extras)
    return next((c for c in cand if c in df.columns), None)

def _build_kinematic_side_mapping(df_mean, wish, side):
    mapping = []
    for nice, (base, extras, mode) in wish.items():
        col = _find_kinematic_col(df_mean, base,
                                   side=side if mode == "sided" else None,
                                   extras=extras)
        if col: mapping.append((col, nice))
    return mapping

def _subset_and_rename_kinematics(df, mapping):
    return pd.DataFrame({nice: df[col] for col, nice in mapping if col in df.columns},
                        index=df.index)

def _plot_lower_body_lumbar_to_file(gaitResults, output_path):
    wish = {
        "extensión lumbar": ("lumbar_extension", ["lumbar_ext"],     "shared"),
        "flexión lumbar":   ("lumbar_bending",   ["lumbar_latbend"], "shared"),
        "rotación lumbar":  ("lumbar_rotation",  ["lumbar_rot"],     "shared"),
        "inclinación pélvica":      ("pelvis_tilt",       [],                 "shared"),
        "desnivel pélvico":      ("pelvis_list",       [],                 "shared"),
        "rotación pélvica":  ("pelvis_rotation",   ["pelvis_rot"],     "shared"),
        "flexión cadera":      ("hip_flexion",        [],                "sided"),
        "adducción cadera":    ("hip_adduction",      [],                "sided"),
        "rotación cadera":     ("hip_rotation",       [],                "sided"),
        "flexión rodilla":     ("knee_flexion",       ["knee_angle"],    "sided"),
        "dorsiflexión tobillo":("ankle_dorsiflexion",["ankle_angle"],   "sided"),
        "ángulo subtalar": ("subtalar_angle",     ["foot_supination"],"sided"),
    }
    dfm_r, dfs_r = gaitResults["curves_r"]["mean"], gaitResults["curves_r"]["sd"]
    dfm_l, dfs_l = gaitResults["curves_l"]["mean"], gaitResults["curves_l"]["sd"]
    map_r  = _build_kinematic_side_mapping(dfm_r, wish, "r")
    labels = [nice for _, nice in map_r]
    if not labels: return
    map_l  = _build_kinematic_side_mapping(dfm_l, wish, "l")
    mr, ml, sr, sl = [_subset_and_rename_kinematics(df, m)
                      for df, m in [(dfm_r,map_r),(dfm_l,map_l),(dfs_r,map_r),(dfs_l,map_l)]]
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), constrained_layout=True, facecolor='white')
    axes = axes.flatten()
    color_r, color_l = "#5e3c99", "#b2ad00"
    for i, lab in enumerate(labels):
        ax = axes[i]; x = np.linspace(0, 100, len(mr.index))
        if lab in mr.columns:
            ax.plot(x, mr[lab], color=color_r, lw=2.0, label="Derecha" if i==0 else "")
            ax.fill_between(x, mr[lab]-sr[lab], mr[lab]+sr[lab], color=color_r, alpha=0.2, lw=0)
        if lab in ml.columns:
            ax.plot(x, ml[lab], color=color_l, lw=2.0, label="Izquierda" if i==0 else "")
            ax.fill_between(x, ml[lab]-sl[lab], ml[lab]+sl[lab], color=color_l, alpha=0.2, lw=0)
        ax.set_ylabel(lab, fontsize=10); ax.set_xlabel("% ciclo de marcha", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6); ax.tick_params(axis='both', labelsize=8)
    for j in range(len(labels), len(axes)): axes[j].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=11)
    fig.savefig(output_path, dpi=450, bbox_inches='tight'); plt.close(fig)

def _plot_kinematic_row_to_file(gaitResults, wish_dict, title, output_path, figsize):
    dfm_r, dfs_r = gaitResults["curves_r"]["mean"], gaitResults["curves_r"]["sd"]
    dfm_l, dfs_l = gaitResults["curves_l"]["mean"], gaitResults["curves_l"]["sd"]
    map_r  = _build_kinematic_side_mapping(dfm_r, wish_dict, "r")
    labels = ([nice for _,nice in map_r] if map_r
              else [nice for _,nice in _build_kinematic_side_mapping(dfm_l, wish_dict, "l")])
    if not labels: return
    map_l = _build_kinematic_side_mapping(dfm_l, wish_dict, "l")
    mr, ml, sr, sl = [_subset_and_rename_kinematics(df, m)
                      for df, m in [(dfm_r,map_r),(dfm_l,map_l),(dfs_r,map_r),(dfs_l,map_l)]]
    fig, axes = plt.subplots(1, len(labels), figsize=figsize, constrained_layout=True, facecolor='white')
    if len(labels) == 1: axes = [axes]
    color_r, color_l = "#5e3c99", "#b2ad00"
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        x_len = len(mr.index) if lab in mr.columns else len(ml.index)
        x = np.linspace(0, 100, x_len)
        if lab in mr.columns:
            ax.plot(x, mr[lab], color=color_r, lw=2.0, label="derecha" if i==0 else "")
            ax.fill_between(x, mr[lab]-sr[lab], mr[lab]+sr[lab], color=color_r, alpha=0.2, lw=0)
        if lab in ml.columns:
            ax.plot(x, ml[lab], color=color_l, lw=2.0, label="izquierda" if i==0 else "")
            ax.fill_between(x, ml[lab]-sl[lab], ml[lab]+sl[lab], color=color_l, alpha=0.2, lw=0)
        ax.set_ylabel(lab, fontsize=10); ax.set_xlabel("% ciclo de marcha", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6); ax.tick_params(axis='both', labelsize=8)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    if title: fig.suptitle(title, fontsize=14, y=1.05)
    fig.savefig(output_path, dpi=450, bbox_inches='tight'); plt.close(fig)

def _plot_pelvis_translations_to_file(gaitResults, output_path):
    wish = {"adelante":  ("pelvis_tx",[],  "shared"),
            "arriba":   ("pelvis_ty",[],  "shared"),
            "derecha":("pelvis_tz",[],  "shared")}
    _plot_kinematic_row_to_file(gaitResults, wish, "Traslaciones del pelvis", output_path, figsize=(12,3.5))

def _plot_arm_angles_to_file(gaitResults, output_path):
    wish = {"flexión del hombro":   ("arm_flex",[],  "sided"),
            "adducción del hombro": ("arm_add", [],  "sided"),
            "rotación del hombro":  ("arm_rot", [],  "sided"),
            "flexión del codo":      ("elbow_flex",[], "sided")}
    _plot_kinematic_row_to_file(gaitResults, wish, "Ángulos de las articulaciones del brazo", output_path, figsize=(15,3.5))

# =========================================================
# ===        HELPERS PARA PDF                           ===
# =========================================================
def _fmt_value(v, n=4):
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, dtype=float).ravel(); arr = arr[np.isfinite(arr)]
        return "nan" if arr.size == 0 else f"{np.mean(arr):.{n}f} (sd={np.std(arr):.{n}f}, n={arr.size})"
    try:
        v = float(v)
        return "nan" if not np.isfinite(v) else f"{v:.{n}f}"
    except Exception:
        return str(v)

def _measure_text_size_ax(ax, s, fontsize=12, fontweight="normal", *, fontname=None, renderer=None):
    fig = ax.figure; canvas = fig.canvas
    if renderer is None:
        if canvas is None or not hasattr(canvas, "get_renderer"):
            FigureCanvasAgg(fig); canvas = fig.canvas
        try:    renderer = canvas.get_renderer()
        except Exception: canvas.draw(); renderer = canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    if ax_bbox.width == 0 or ax_bbox.height == 0: return 0.0, 0.0
    fp_kwargs = {"size": fontsize, "weight": fontweight}
    if fontname: fp_kwargs["family"] = fontname
    fp = FontProperties(**fp_kwargs)
    w, h, d = renderer.get_text_width_height_descent(s, fp, ismath=False)
    return w / ax_bbox.width, (h + d) / ax_bbox.height

def _consume_words_for_width(ax, words_deque, limit_w, *, fontsize, fontname=None, renderer=None, tolerance=0.04):
    if limit_w <= 0: return []
    from collections import deque
    collected = []
    while words_deque:
        candidate = collected + [words_deque[0]]
        txt = " ".join(candidate)
        width, _ = _measure_text_size_ax(ax, txt, fontsize=fontsize, fontname=fontname, renderer=renderer)
        if width <= limit_w * (1.0 + tolerance) or not collected:
            collected.append(words_deque.popleft())
        else: break
    return collected

def draw_bullet_paragraph(ax, title, desc, *, x=0.07, y=0.885, width=0.8, fs_text=10,
                           bullet="• ", title_suffix=" ", line_spacing_mult=1.12,
                           indent_min=0.03, after_paragraph_gap_mult=0.65,
                           fontname="Cambria", wrap_tolerance=0.03):
    from collections import deque
    renderer = None
    if ax.figure.canvas:
        try: renderer = ax.figure.canvas.get_renderer()
        except Exception:
            FigureCanvasAgg(ax.figure); ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
    desc = (desc or "").strip()
    title_txt = bullet + title + (":" if desc else "") + title_suffix
    ax.text(x, y, title_txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=fs_text, fontweight="bold", fontname=fontname, color="black")
    line_height = _measure_text_size_ax(ax, "Ag", fontsize=fs_text, fontname=fontname, renderer=renderer)[1]
    if line_height == 0: line_height = 0.03
    step = line_height * line_spacing_mult
    current_y = y - step
    words = deque(desc.split()); lines_drawn = 0
    while words:
        next_line_words = _consume_words_for_width(ax, words, width - indent_min,
                                                    fontsize=fs_text, fontname=fontname,
                                                    renderer=renderer, tolerance=wrap_tolerance)
        if not next_line_words: break
        ax.text(x + indent_min, current_y, " ".join(next_line_words),
                transform=ax.transAxes, ha="left", va="top", fontsize=fs_text,
                fontname=fontname, color="black")
        current_y -= step; lines_drawn += 1
    return y - step * (1 + lines_drawn) - step * after_paragraph_gap_mult

def build_bullets(patient):
    altura = float(patient.get("altura") or 0.0)
    sl_thresh = 0.45 * altura if altura > 0 else None
    stride_txt = (
        f"Corresponde la distancia entre las posiciones del calcáneo (talón) "
        f"al principio y al final del ciclo de marcha. Una longitud de zancada superior a 0.45 veces la altura del sujeto "
        f"se considera buena. (> {_fmt_value(sl_thresh)} m para este sujeto)." if sl_thresh
        else "Corresponde la distancia entre las posiciones del calcáneo (talón) al principio "
             "y al final del ciclo de marcha. Una longitud de zancada superior a 0.45 veces la altura del sujeto se considera buena."
    )
    return [
        ("Cadencia", "La cadencia se calcula como el número de ciclos de marcha (izquierdo y derecho) por minuto. Una cadencia superior a 100 se considera buena."),
        ("Longitud de zancada", stride_txt),
        ("Ancho del paso", "Corresponde a la distancia promedio entre los centros de las articulaciones del tobillo en dirección mediolateral durante el 40-60% de la fase de apoyo. Un ancho de paso entre 4.3 y 7.4 veces la altura del sujeto se considera bueno."),
        ("Velocidad de la marcha", "Se calcula dividiendo el desplazamiento del centro de masa por el tiempo que tarda en recorrer esa distancia. Una velocidad superior a 1.12 m/s se considera buena."),
        ("Valgo dinámico", "Durante apoyo medio corresponde al ángulo máximo entre el 10–30% del ciclo. Ángulo positivo: valgo; negativo: varo. Un valor entre −5 y 5° se considera adecuado."),
        ("Amplitud del pie", "Corresponde a la altura mínima entre el pie (dedo o talón) y el suelo durante la oscilación. Valores cercanos a 20–25 mm se consideran adecuados en población sana."),
    ]

def add_centered_info_table(fig, items, center_y=0.815, **kwargs):
    opts = {"width_frac": 0.85, "row_height": 0.02, "fontsize": 10,
            "fontname": "Cambria", "lw": 0.8, "edge": "black", **kwargs}
    ax = fig.add_axes([0.5 - opts['width_frac']/2.0,
                       center_y - opts['row_height']/2.0,
                       opts['width_frac'], opts['row_height']])
    ax.axis("off")
    tbl = ax.table(cellText=[items], cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(opts['fontsize'])
    for cell in tbl.get_celld().values():
        cell.set_edgecolor(opts['edge']); cell.set_linewidth(opts['lw'])
        if cell.get_text(): cell.get_text().set_fontname(opts['fontname'])

def export_pdf_report(save_path, logo_path, patient, trial_name, fig_results, metrics,
                       curves_r, curves_l, fall_risk_result=None):
    a4 = (8.27, 11.69)
    def footer_page_number(fig, n):
        fig.text(0.975, 0.02, f"{n}", ha="right", va="bottom", fontsize=9, color="#444")

    canvas_refs = []
    with PdfPages(save_path) as pdf, tempfile.TemporaryDirectory() as tmpdir:

        # --- Página 1: Portada y Barras ---
        page_no = 1

        # Guardar barras como PNG con recorte real de espacio blanco usando numpy
        path_bars_png = os.path.join(tmpdir, "bars.png")
        fig_results.savefig(path_bars_png, dpi=450, bbox_inches='tight',
                            pad_inches=0.05, facecolor="white")
        _img_full = plt.imread(path_bars_png)
        # Recortar filas blancas superior e inferior
        _rgb8 = (_img_full[:, :, :3] * 255).astype(np.uint8) if _img_full.max() <= 1.0 else _img_full[:, :, :3].astype(np.uint8)
        _nonwhite = np.where(~np.all(_rgb8 > 245, axis=(1, 2)))[0]
        if _nonwhite.size > 0:
            _t = max(0, _nonwhite[0] - 3)
            _b = min(_img_full.shape[0], _nonwhite[-1] + 3)
            _img_cropped = _img_full[_t:_b, :, :]
        else:
            _img_cropped = _img_full

        fig1 = Figure(figsize=a4, facecolor='white'); canvas_refs.append(FigureCanvasAgg(fig1))
        try:
            if os.path.isfile(logo_path):
                logo_img = load_logo_as_array(logo_path, max_width_px=1000)
                logo_ax  = fig1.add_axes([0.07, 0.86, 0.18, 0.12])
                logo_ax.imshow(logo_img); logo_ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not load logo: {e}")
        fig1.text(0.5, 0.86, "Informe de Análisis de Marcha", ha="center", va="center",
                  fontsize=20, fontweight="bold", fontname="Cambria")
        patient_info = [f"Paciente: {patient.get('nombre','')}", f"Edad: {patient.get('edad','')} años",
                        f"Altura: {patient.get('altura','')} m", f"Modo: {str(patient.get('modo','')).capitalize()}"]
        add_centered_info_table(fig1, patient_info, 0.82)
        fig1.text(0.08, 0.755, "1. Parámetros Espacio-Temporales", fontsize=14,
                  fontweight="bold", fontname="Cambria")
        # Calcular altura proporcional del eje imagen para que no quede aplastada
        _ih, _iw = _img_cropped.shape[:2]
        # El eje ocupa ancho=0.90 en la página A4 (8.27in) → ancho real ≈ 7.44in
        # Calcular la altura proporcional en fracción de página (11.69in)
        _ax_w_in = 0.90 * 8.27
        _ax_h_in = _ax_w_in * (_ih / _iw)
        _ax_h_frac = min(0.70, _ax_h_in / 11.69)  # no superar 70% de la página
        _ax_bottom = 0.73 - _ax_h_frac  # justo debajo del título
        _ax_bottom = max(0.03, _ax_bottom)
        ax_img = fig1.add_axes([0.05, _ax_bottom, 0.90, _ax_h_frac])
        ax_img.imshow(_img_cropped, interpolation="none", aspect="auto")
        ax_img.axis('off')
        footer_page_number(fig1, page_no); pdf.savefig(fig1, dpi=300); plt.close(fig1)

        # --- Página 2: Descripciones ---
        page_no += 1
        fig2 = Figure(figsize=a4, facecolor='white'); canvas_refs.append(FigureCanvasAgg(fig2))
        ax2  = fig2.add_axes([0.03,0,1,1]); ax2.axis('off'); fig2.canvas.draw()
        fig2.text(0.07, 0.94, "Descripción de los Parámetros", fontsize=12,
                  fontweight="bold", fontname="Cambria")
        y_pos = 0.92
        for title, desc in build_bullets(patient):
            if y_pos < 0.2:
                footer_page_number(fig2, page_no); pdf.savefig(fig2, dpi=300); plt.close(fig2)
                page_no += 1
                fig2 = Figure(figsize=a4, facecolor='white'); canvas_refs.append(FigureCanvasAgg(fig2))
                ax2  = fig2.add_axes([0.03,0,1,1]); ax2.axis('off'); fig2.canvas.draw()
                fig2.text(0.07, 0.94, "Descripción de los Parámetros", fontsize=12,
                          fontweight="bold", fontname="Cambria")
                y_pos = 0.92
            y_pos = draw_bullet_paragraph(ax2, title, desc, y=y_pos)
        footer_page_number(fig2, page_no); pdf.savefig(fig2, dpi=300); plt.close(fig2)

        gaitResults = {"curves_r": curves_r, "curves_l": curves_l}

        # --- Página 3: Cinemática Lower-body ---
        page_no += 1
        fig3 = Figure(figsize=a4, facecolor='white'); canvas_refs.append(FigureCanvasAgg(fig3))
        fig3.text(0.07, 0.96, "2. Cinemática articular", fontname="Cambria", fontsize=14,
                  fontweight="bold", ha="left", va="top")
        fig3.text(0.085, 0.92, "2.1. Ángulos de las articulaciones lumbares y de la parte inferior del cuerpo",
                  fontname="Cambria", fontsize=12, fontweight="bold", ha="left", va="top")
        path_img1 = os.path.join(tmpdir, "lower_body.png")
        _plot_lower_body_lumbar_to_file(gaitResults, path_img1)
        if os.path.exists(path_img1):
            ax_img1 = fig3.add_axes([0.05, 0.26, 0.9, 0.8])
            ax_img1.imshow(plt.imread(path_img1), interpolation="none"); ax_img1.axis('off')
        definitions = [
            "• Extensión lumbar (plano sagital) es positiva cuando el torso rota hacia atrás.",
            "• Inclinación lumbar (plano frontal) es positiva cuando el hombro izquierdo se eleva.",
            "• Rotación lumbar (plano transversal) es positiva cuando el torso rota hacia la izquierda.",
            "• Inclinación pélvica (tilt) es positiva cuando la pelvis rota hacia atrás.",
            "• Oblicuidad pélvica (list) es positiva cuando el lado izquierdo de la pelvis se eleva.",
            "• Rotación pélvica es positiva cuando la pelvis rota hacia la izquierda.",
            "• Rotación de cadera es positiva cuando la pierna rota hacia adentro (rotación interna).",
        ]
        y_pos_text = 0.4
        for text in definitions:
            fig3.text(0.12, y_pos_text, text, fontname="Cambria", fontsize=10, ha="left", va="top")
            y_pos_text -= 0.035
        footer_page_number(fig3, page_no); pdf.savefig(fig3, dpi=300); plt.close(fig3)

        # --- Página 4: Pelvis y Brazos ---
        page_no += 1
        fig4 = Figure(figsize=a4, facecolor='white'); canvas_refs.append(FigureCanvasAgg(fig4))
        fig4.text(0.07, 0.94, "2.2. Pelvis", fontsize=12, fontweight="bold", fontname="Cambria")
        path_img_pelvis = os.path.join(tmpdir, "pelvis.png")
        _plot_pelvis_translations_to_file(gaitResults, path_img_pelvis)
        if os.path.exists(path_img_pelvis):
            ax_img2 = fig4.add_axes([0.05, 0.65, 0.9, 0.35])
            ax_img2.imshow(plt.imread(path_img_pelvis), interpolation="none"); ax_img2.axis('off')
        fig4.text(0.07, 0.68, "2.3. Ángulos de las articulaciones del brazo",
                  fontsize=12, fontweight="bold", fontname="Cambria")
        path_img_arms = os.path.join(tmpdir, "arms.png")
        _plot_arm_angles_to_file(gaitResults, path_img_arms)
        if os.path.exists(path_img_arms):
            ax_img3 = fig4.add_axes([0.05, 0.405, 0.9, 0.35])
            ax_img3.imshow(plt.imread(path_img_arms), interpolation="none"); ax_img3.axis('off')
        fig4.text(0.12, 0.48, "• Rotación de hombro es positiva cuando el brazo rota hacia adentro (rotación interna).",
                  fontname="Cambria", fontsize=10, ha="left", va="top")
        footer_page_number(fig4, page_no); pdf.savefig(fig4, dpi=300); plt.close(fig4)

        # --- Páginas 5 y 6: Riesgo de Caída completo (2 hojas) ---
        if fall_risk_result:
            _fr           = fall_risk_result.get("fall_risk", {})
            _gs           = fall_risk_result.get("gait_stats", {})
            _ent          = fall_risk_result.get("entropies", {})
            _level        = str(_fr.get("risk_level", "ND")).upper()
            _sb = _fr.get("scorebase")
            _sf = _fr.get("scorefinal")
            _sse = _fr.get("scoresampen")
            _sauc = _fr.get("scoreaucmse")

            _s_ent_total = None
            if _sse is not None and _sauc is not None:
                try:
                    _s_ent_total = float(_sse) + float(_sauc)
                except (TypeError, ValueError):
                    _s_ent_total = None

            _s_combined = None
            if _sb is not None and _s_ent_total is not None:
                try:
                    _s_combined = 0.60 * float(_sb) + 0.40 * float(_s_ent_total)
                except (TypeError, ValueError):
                    _s_combined = None

            def _ent_risk_level(_score):
                if _score is None:
                    return "N/D", "#333333"
                if _score >= 60:
                    return "ALTO", "#c0392b"
                if _score >= 30:
                    return "MODERADO", "#e67e22"
                return "BAJO", "#27ae60"

            _ent_level, _ent_color = _ent_risk_level(_s_ent_total)
            _comb_level, _comb_color = _ent_risk_level(_s_combined)
            _color_map    = {"ALTO": "#c0392b", "MODERADO": "#e67e22", "BAJO": "#27ae60"}
            _bg_map       = {"ALTO": "#fdf3f2", "MODERADO": "#fef9f2", "BAJO": "#f2faf5"}
            _risk_color   = _color_map.get(_level, "#555555")
            _bg_color     = _bg_map.get(_level,    "#f9f9f9")
            _interpretation = _fr.get("interpretation", "")
            _combined_exp   = _fr.get("combined_interpretation", "")
            _entropy_note   = _fr.get("entropy_note", "")
            _n_trials       = fall_risk_result.get("n_trials", 1)
            _sb  = _fr.get("score_base")
            _sf  = _fr.get("score_final")
            _sse = _fr.get("score_sampen")
            _sauc= _fr.get("score_aucmse")
            # Score entropía combinado y ponderado
            _s_ent_total = None
            if _sse is not None and _sauc is not None:
                try:
                    _s_ent_total = float(_sse) + float(_sauc)
                except (TypeError, ValueError):
                    _s_ent_total = None

            _s_combined = None
            if _sb is not None and _s_ent_total is not None:
                try:
                    _s_combined = 0.60 * float(_sb) + 0.40 * float(_s_ent_total)
                except (TypeError, ValueError):
                    _s_combined = None

            def _ent_risk_level(score):
                if score is None: return "N/D", "#333333"
                if score >= 60:   return "ALTO",     "#c0392b"
                if score >= 30:   return "MODERADO", "#e67e22"
                return "BAJO", "#27ae60"

            _ent_level, _ent_color   = _ent_risk_level(_s_ent_total)
            _comb_level, _comb_color = _ent_risk_level(_s_combined)
            _fh  = _fr.get("fall_history_applied", False)
            _contributing = _fr.get("contributing_factors", [])
            _pname = patient.get("nombre", "")
            _edad  = patient.get("edad", "")

            def _fe2(v):
                try:
                    f = float(v)
                    return f"{f:.0f}" if not _math.isnan(f) else "N/A"
                except Exception:
                    return "N/A"

            def _fmt2(v):
                try:
                    f = float(v)
                    return f"{f:.3f}" if not _math.isnan(f) else "N/A"
                except Exception:
                    return "N/A"

            def _wraplines(text, width=100):
                return _tw.wrap(str(text), width=width) or [str(text)]

            def _hdiv_ent(fig, y):
                ax = fig.add_axes([0.07, y, 0.86, 0.001])
                ax.set_facecolor("#dddddd"); ax.axis("off")
                return y - 0.018

            def _sec_title_ent(fig, y, text):
                fig.text(0.07, y, text, fontname="Cambria", fontsize=11,
                         fontweight="bold", color="#2c3e50", ha="left", va="top")
                return y - 0.032
            
            def _cat_sampen(_val, _signal="stride_time"):
                try:
                    _v = float(_val)
                except Exception:
                    return "N/D"

                if _signal == "stride_time":
                    if _v < 1.00 or _v > 1.70:
                        return "patológico"
                    elif (1.00 <= _v < 1.20) or (1.50 < _v <= 1.70):
                        return "limítrofe"
                    else:
                        return "normal"

                if _signal == "stride_time_variability":
                    if _v < 1.00 or _v > 2.50:
                        return "patológico"
                    elif (1.00 <= _v < 1.20) or (2.00 < _v <= 2.50):
                        return "limítrofe"
                    else:
                        return "normal"

                return "N/D"

            def _cat_aucmse(_val, _signal="stride_time"):
                try:
                    _v = float(_val)
                except Exception:
                    return "N/D"

                if _signal == "stride_time":
                    if _v < 0.82 or _v > 1.28:
                        return "patológico"
                    elif (0.82 <= _v < 0.92) or (1.15 < _v <= 1.28):
                        return "limítrofe"
                    else:
                        return "normal"

                if _signal == "trunk_ap_acceleration":
                    if _v < 0.78 or _v > 1.25:
                        return "patológico"
                    elif (0.78 <= _v < 0.88) or (1.12 < _v <= 1.25):
                        return "limítrofe"
                    else:
                        return "normal"

                return "N/D"

            # ── HOJA 5: scores + interpretación ──────────────────────────────
            page_no += 1
            _fig5 = Figure(figsize=a4, facecolor="white")
            canvas_refs.append(FigureCanvasAgg(_fig5))

            _fig5.text(0.07, 0.96, "3. Riesgo de Caída",
                       fontname="Cambria", fontsize=14, fontweight="bold",
                       color="#2c3e50", ha="left", va="top")

            _yt = 0.88
            _yt = _hdiv_ent(_fig5, _yt)

            _banner = _fig5.add_axes([0.07, 0.81, 0.86, 0.11])
            _banner.set_facecolor("#eef4f1")
            for _sp in _banner.spines.values():
                _sp.set_edgecolor("#e1e8e4")
                _sp.set_linewidth(0.8)
            _banner.set_xticks([])
            _banner.set_yticks([])
            _banner.set_xlim(0, 1)
            _banner.set_ylim(0, 1)

            _clin_level, _clin_color = _ent_risk_level(_sf if _sf is not None else _sb)

            _score_clinico_txt = f"{float(_sb):.1f}/100" if _sb is not None else "N/D"
            _score_entropia_txt = f"{float(_s_ent_total):.1f}/100" if _s_ent_total is not None else "N/D"
            _score_comb_txt = f"{float(_s_combined):.1f}/100" if _s_combined is not None else "N/D"
            _score_sampen_txt = f"{float(_sse):.0f}/50" if _sse is not None else "N/D"
            _score_auc_txt = f"{float(_sauc):.0f}/50" if _sauc is not None else "N/D"

            _banner.text(0.02, 0.75, "●", color=_clin_color, fontsize=10, va="center", ha="left")
            _banner.text(0.055, 0.75, f"Riesgo {_clin_level}", color=_clin_color,
                         fontname="Cambria", fontsize=12, fontweight="bold", va="center", ha="left")
            _banner.text(0.3, 0.75, f"Score clínico: {_score_clinico_txt}", color="#555555",
                         fontname="Cambria", fontsize=9.5, va="center", ha="left")

            _banner.text(0.02, 0.48, "●", color=_ent_color, fontsize=10, va="center", ha="left")
            _banner.text(0.055, 0.48, f"Riesgo {_ent_level}", color=_ent_color,
                         fontname="Cambria", fontsize=12, fontweight="bold", va="center", ha="left")
            _banner.text(0.3, 0.48, f"Score Entropía: {_score_entropia_txt}", color="#555555",
                         fontname="Cambria", fontsize=9.5, va="center", ha="left")

            _banner.text(0.55, 0.53, f"Score SampEn: {_score_sampen_txt}", color="#666666",
                         fontname="Cambria", fontsize=8.5, va="center", ha="left")
            _banner.text(0.55, 0.43, f"Score AUC-MSE: {_score_auc_txt}", color="#666666",
                         fontname="Cambria", fontsize=8.5, va="center", ha="left")
            
            _banner.text(0.02, 0.20, "●", color=_comb_color, fontsize=10, va="center", ha="left")
            _banner.text(0.055, 0.20, f"Riesgo Combinado (60/40): {_comb_level}", color=_comb_color,
                         fontname="Cambria", fontsize=11.5, fontweight="bold", va="center", ha="left")
            _banner.text(0.53, 0.20, f"Score: {_score_comb_txt}", color="#555555",
                         fontname="Cambria", fontsize=9.5, va="center", ha="left")
            
            _yt = 0.8
            _yt = _hdiv_ent(_fig5, _yt)
            _yt = _sec_title_ent(_fig5, _yt, "Estadísticas de marcha")
            _stats_rows = [
                ("stride_time_mean_s",    None,  "stride_time_cv_pct",   "Tiempo de zancada",  "s"),
                ("stride_length_mean_m",  None,                "stride_length_cv_pct", "Longitud de zancada",   "m"),
                ("cadence_mean_spm",      None,                "cadence_cv_pct",        "Cadencia",           "pasos/min"),
                ("walking_speed_mean_ms", None,                None,                   "Velocidad de marcha","m/s"),
            ]
            for _mk, _sdk, _cvk, _lbl, _unit in _stats_rows:
                _mv = _gs.get(_mk)
                if _mv is None: continue
                try:
                    _mv  = float(_mv)
                    _sdv = float(_gs[_sdk]) if _sdk and _gs.get(_sdk) is not None else None
                    _cvv = float(_gs[_cvk]) if _cvk and _gs.get(_cvk) is not None else None
                    _fmt_s = ".1f" if _mk == "cadence_mean_spm" else ".3f"
                    _fig5.text(0.09, _yt, f"{_lbl}:  {_mv:{_fmt_s}} {_unit}",
                               fontname="Cambria", fontsize=10, color="#1a5276", ha="left", va="top")
                    if _sdv is not None:
                        _fig5.text(0.46, _yt, f"SD: {_sdv:.3f}",
                                   fontname="Cambria", fontsize=10, color="#1a5276", ha="left", va="top")
                    if _cvv is not None:
                        _fig5.text(0.63, _yt, f"CV: {_cvv:.2f}%",
                                   fontname="Cambria", fontsize=10, color="#1a5276", ha="left", va="top")
                    _yt -= 0.025
                except Exception:
                    pass
            _yt -= 0.001
            _yt = _hdiv_ent(_fig5, _yt)

            _yt -= 0.005
            _yt = _sec_title_ent(_fig5, _yt, "Interpretación clínica")

            # Párrafo 1 — nivel de riesgo
            _interp_text = _interpretation
            if not _interp_text:
                if _level == "ALTO":
                    _interp_text = (
                        "El perfil cinemático de este paciente presenta múltiples indicadores "
                        "compatibles con riesgo elevado de caída. El patrón global sugiere "
                        "alteraciones relevantes en velocidad, variabilidad y consistencia de la marcha, "
                        "por lo que se recomienda valoración clínica y seguimiento estrecho."
                    )
                elif _level == "MODERADO":
                    _interp_text = (
                        "El perfil cinemático muestra señales de alerta en algunos parámetros de marcha. "
                        "El patrón global es compatible con riesgo intermedio, por lo que se recomienda "
                        "seguimiento clínico periódico y reevaluación funcional."
                    )
                else:
                    _interp_text = (
                        "El perfil cinemático se encuentra globalmente dentro de rangos conservados. "
                        "El patrón observado es compatible con bajo riesgo clínico de caída, "
                        "sin perjuicio de mantener seguimiento preventivo habitual."
                    )
            for _line in _wraplines(_interp_text, 120):
                if _yt < 0.06: break
                _fig5.text(0.09, _yt, _line, fontname="Cambria", fontsize=9.5,
                           color="#000000", style="italic", ha="left", va="top")
                _yt -= 0.025

            # Párrafo 2 — factores contribuyentes
            if _contributing:
                _yt -= 0.006
                _fig5.text(0.09, _yt, "Factores clínicos detectados y su significado:",
                           fontname="Cambria", fontsize=9.5, color="#000000",
                           style="italic", ha="left", va="top")
                _yt -= 0.025

                _p2_map = [
                    (
                        ("cv stride time", "variabilidad", "stride time cv"),
                        "· CV de tiempo de zancada elevado: mayor variabilidad en el tiempo de zancada es uno de los "
                        "predictores más robustos de caídas en adultos mayores. Refleja pérdida de regulación fina del "
                        "ritmo locomotor y menor estabilidad dinámica."
                    ),
                    (
                        ("velocidad", "speed", "gait speed"),
                        "· Velocidad de marcha reducida: la velocidad es la variable integradora más potente del estado "
                        "funcional. Valores bajos se asocian a mayor fragilidad, discapacidad, mortalidad y riesgo de caída."
                    ),
                    (
                        ("stride length", "longitud", "longitud de zancada"),
                        "· Longitud de zancada reducida: una zancada corta sugiere estrategia de marcha más cautelosa y "
                        "menor capacidad de propulsión. Cuando además es variable, se asocia a mayor riesgo de caída."
                    ),
                    (
                        ("cadencia", "cadence"),
                        "· Alteración en cadencia: cadencias demasiado bajas o altas, o con oscilación aumentada entre "
                        "pasos, se asocian a inestabilidad y a menor capacidad de sostener un ritmo locomotor estable."
                    ),
                    (
                        ("cv stride length", "variabilidad longitud", "stride length cv"),
                        "· CV de longitud de zancada elevado: indica pérdida de consistencia en el patrón espacial de la "
                        "marcha y menor capacidad de mantener un paso uniforme entre ciclos."
                    ),
                    (
                        ("step width", "ancho del paso", "step width variability", "variabilidad del ancho"),
                        "· Alteración del ancho del paso o de su variabilidad: puede reflejar una estrategia compensatoria "
                        "para aumentar estabilidad lateral, pero cuando es excesiva se asocia a peor control mediolateral."
                    ),
                    (
                        ("double support", "doble apoyo"),
                        "· Aumento del tiempo de doble apoyo: suele interpretarse como una estrategia compensatoria de "
                        "seguridad, compatible con menor confianza para transferir carga de un miembro al otro."
                    ),
                    (
                        ("single support", "apoyo unipodal", "single support"),
                        "· Disminución del tiempo de apoyo unipodal: sugiere menor tolerancia a la carga sobre una sola "
                        "extremidad y menor estabilidad durante la fase media de apoyo."
                    ),
                    (
                        ("symmetry", "simetr", "asimetr"),
                        "· Asimetría entre lados: diferencias relevantes entre miembros sugieren un patrón menos eficiente "
                        "y menos automático, compatible con compensaciones motoras."
                    ),
                ]

                _used_p2 = set()
                _p2_parts = []

                for _fi in _contributing:
                    _fi_low = str(_fi).lower()
                    for _keys, _msg in _p2_map:
                        if _msg in _used_p2:
                            continue
                        if any(_k in _fi_low for _k in _keys):
                            _p2_parts.append(_msg)
                            _used_p2.add(_msg)
                            break

                if _p2_parts:
                    for _p2 in _p2_parts:
                        for _line in _wraplines(_p2, 120):
                            if _yt < 0.06:
                                break
                            _fig5.text(
                                0.09, _yt, _line,
                                fontname="Cambria", fontsize=9.5,
                                color="#000000", style="italic",
                                ha="left", va="top"
                            )
                            _yt -= 0.025

                        _yt -= 0.01

            # Párrafo 3 — análisis de complejidad
            _ent_st = _ent.get("stride_time", {})
            _ent_stv = _ent.get("stride_time_variability", {})
            _ent_ta = _ent.get("trunk_ap_acceleration", {})

            _p3_parts = []

            _se_st = _ent_st.get("sampen")
            if _se_st is not None:
                _cat = _cat_sampen(_se_st, "stride_time")
                if _cat == "normal":
                    _p3_parts.append(
                        f"· SampEn de tiempo de zancada ({_se_st:.3f}): patrón compatible con complejidad preservada."
                    )
                elif _cat == "limítrofe":
                    _p3_parts.append(
                        f"· SampEn de tiempo de zancada ({_se_st:.3f}): patrón limítrofe, compatible con cambios leves en la regularidad del control locomotor."
                    )
                elif _cat == "patológico":
                    _p3_parts.append(
                        f"· SampEn de tiempo de zancada ({_se_st:.3f}): patrón patológico, compatible con pérdida de adaptabilidad del sistema locomotor."
                    )

            _se_stv = _ent_stv.get("sampen")
            if _se_stv is not None:
                _cat = _cat_sampen(_se_stv, "stride_time_variability")
                if _cat == "normal":
                    _p3_parts.append(
                        f"· SampEn de la variabilidad local del tiempo de zancada ({_se_stv:.3f}): patrón compatible con organización temporal preservada."
                    )
                elif _cat == "limítrofe":
                    _p3_parts.append(
                        f"· SampEn de la variabilidad local del tiempo de zancada ({_se_stv:.3f}): patrón limítrofe, con irregularidad discreta del ritmo de marcha."
                    )
                elif _cat == "patológico":
                    _p3_parts.append(
                        f"· SampEn de la variabilidad local del tiempo de zancada ({_se_stv:.3f}): patrón patológico, sugerente de desorganización del control temporal de la marcha."
                    )

            _auc_st = _ent_st.get("mse", {}).get("auc_mse")
            if _auc_st is not None:
                _cat = _cat_aucmse(_auc_st, "stride_time")
                if _cat == "normal":
                    _p3_parts.append(
                        f"· AUC-MSE del tiempo de zancada ({_auc_st:.3f}): patrón compatible con complejidad multiescala preservada."
                    )
                elif _cat == "limítrofe":
                    _p3_parts.append(
                        f"· AUC-MSE del tiempo de zancada ({_auc_st:.3f}): patrón limítrofe, con reducción parcial de la complejidad multiescala."
                    )
                elif _cat == "patológico":
                    _p3_parts.append(
                        f"· AUC-MSE del tiempo de zancada ({_auc_st:.3f}): patrón patológico, compatible con pérdida de complejidad multiescala."
                    )

            _auc_ta = _ent_ta.get("mse", {}).get("auc_mse")
            if _auc_ta is not None:
                _cat = _cat_aucmse(_auc_ta, "trunk_ap_acceleration")
                if _cat == "normal":
                    _p3_parts.append(
                        f"· AUC-MSE de la aceleración AP del tronco ({_auc_ta:.3f}): patrón compatible con control dinámico preservado."
                    )
                elif _cat == "limítrofe":
                    _p3_parts.append(
                        f"· AUC-MSE de la aceleración AP del tronco ({_auc_ta:.3f}): patrón limítrofe, compatible con cambios leves en la estabilidad dinámica."
                    )
                elif _cat == "patológico":
                    _p3_parts.append(
                        f"· AUC-MSE de la aceleración AP del tronco ({_auc_ta:.3f}): patrón patológico, sugerente de alteración en la complejidad del control postural durante la marcha."
                    )

            if _p3_parts:
                _yt -= 0.006
                _fig5.text(0.09, _yt, "Análisis de complejidad de la marcha:",
                           fontname="Cambria", fontsize=9.5, color="#000000",
                           style="italic", ha="left", va="top")
                _yt -= 0.025
                for _p3 in _p3_parts:
                    for _line in _wraplines(_p3, 120):
                        if _yt < 0.06: break
                        _fig5.text(0.09, _yt, _line, fontname="Cambria", fontsize=9.5,
                                   color="#000000", style="italic", ha="left", va="top")
                        _yt -= 0.025
            _yt -= 0.01
            _yt = _hdiv_ent(_fig5, _yt)
            _yt = _sec_title_ent(_fig5, _yt, "Interpretación combinada")

            if _combined_exp:
                _combined_text = _combined_exp
            else:
                _parts = []
                _p2_map = [
                    (
                        ("cv tiempo de zancada",),
                        "· CV de tiempo de zancada elevado: mayor variabilidad en el tiempo de zancada es uno de los "
                        "predictores más robustos de caídas en adultos mayores. Refleja pérdida de regulación fina del "
                        "ritmo locomotor y menor estabilidad dinámica."
                    ),
                    (
                        ("velocidad",),
                        "· Velocidad de marcha reducida: la velocidad es la variable integradora más potente del estado "
                        "funcional. Valores bajos se asocian a mayor fragilidad, discapacidad, mortalidad y riesgo de caída."
                    ),
                    (
                        ("cv longitud de zancada",),
                        "· CV de longitud de zancada elevado: indica pérdida de consistencia en el patrón espacial de la "
                        "marcha y menor capacidad de mantener un paso uniforme entre ciclos."
                    ),
                    (
                        ("longitud de zancada",),
                        "· Longitud de zancada reducida: una zancada corta sugiere estrategia de marcha más cautelosa y "
                        "menor capacidad de propulsión. Cuando además es variable, se asocia a mayor riesgo de caída."
                    ),
                    (
                        ("cv cadencia",),
                        "· CV de cadencia elevado: alta variabilidad en cadencia se asocia a inestabilidad y menor "
                        "capacidad de sostener un ritmo locomotor estable (Lord 2011)."
                    ),
                ]
                if _contributing:
                    for _fi in _contributing:
                        _fi_low = _fi.lower()
                        for _keys, _msg in _p2_map:
                            if any(_k in _fi_low for _k in _keys):
                                _parts.append(_msg); break
                _ent_st = _ent.get("stride_time", {})
                if _ent_st:
                    _se_v  = _ent_st.get("sampen")
                    _mse_v = _ent_st.get("mse", {}).get("auc_mse")
                    if _se_v is not None:
                        _cat = _cat_sampen(_se_v, "stride_time")
                        if _cat == "patológico":
                            _parts.append(f"- SampEn stride time ({_se_v:.3f}) fuera de rango (1.20-1.50): patrón patológico (Stergiou 2011).")
                        elif _cat == "limítrofe":
                            _parts.append(f"- SampEn stride time ({_se_v:.3f}) limítrofe: alerta leve en complejidad (Stergiou 2011).")
                        else:
                            _parts.append(f"- SampEn stride time ({_se_v:.3f}) en rango normal (1.20-1.50): complejidad preservada.")
                    if _mse_v is not None:
                        _cat = _cat_aucmse(_mse_v, "stride_time")
                        if _cat == "patológico":
                            _parts.append(f"- AUC-MSE ({_mse_v:.3f}) fuera de rango (0.92-1.15): pérdida de complejidad multiescala (Lipsitz 2004).")
                        elif _cat == "limítrofe":
                            _parts.append(f"- AUC-MSE ({_mse_v:.3f}) limítrofe: reducción parcial de complejidad multiescala.")
                        else:
                            _parts.append(f"- AUC-MSE ({_mse_v:.3f}) en rango normal (0.92-1.15): complejidad multiescala preservada.")
                _combined_text = "\n".join(_parts)

            for _raw_line in _combined_text.split("\n"):
                for _line in _wraplines(_raw_line.strip(), 120):
                    if _yt < 0.06: break
                    _fig5.text(0.09, _yt, _line, fontname="Cambria", fontsize=9.5,
                               color="#000000", style="italic", ha="left", va="top")
                    _yt -= 0.025

            footer_page_number(_fig5, page_no)
            pdf.savefig(_fig5, dpi=300)
            plt.close(_fig5)

            # ── HOJA 6: factores + stats + entropías + glosario ──────────────
            page_no += 1
            _fig6 = Figure(figsize=a4, facecolor="white")
            canvas_refs.append(FigureCanvasAgg(_fig6))
            _yt2 = 0.945

            _yt2 = _hdiv_ent(_fig6, _yt2)
            _yt2 = _sec_title_ent(_fig6, _yt2, "Factores clínicos contribuyentes")
            if _contributing:
                for _factor in _contributing:
                    _fig6.text(0.09, _yt2, f"- {_factor}",
                               fontname="Cambria", fontsize=10, color="#7b241c", ha="left", va="top")
                    _yt2 -= 0.025
            else:
                _fig6.text(0.09, _yt2, "Sin factores de riesgo destacados.",
                           fontname="Cambria", fontsize=10, color="#999999",
                           style="italic", ha="left", va="top")
                _yt2 -= 0.025
            _yt2 -= 0.01
            _yt2 = _hdiv_ent(_fig6, _yt2)

            _yt2 = _sec_title_ent(_fig6, _yt2, "Entropías descriptivas")
            _ent_rows = [
                ("stride_time", "Tiempo de zancada"),
                ("stride_time_variability", "Variabilidad del tiempo de zancada"),
                ("trunk_ap_acceleration", "Aceleración AP del tronco"),
            ]
            for _ekey, _elab in _ent_rows:
                _e   = _ent.get(_ekey, {})
                _se  = _e.get("sampen",  float("nan"))
                _auc = _e.get("mse", {}).get("auc_mse", float("nan"))
                _npt = _e.get("n", 0)
                _fig6.text(0.09, _yt2, f"{_elab}:",
                           fontname="Cambria", fontsize=10, color="#444444", ha="left", va="top")
                _fig6.text(0.38, _yt2, f"SampEn: {_fmt2(_se)}",
                           fontname="Cambria", fontsize=10, color="#444444", ha="left", va="top")
                _fig6.text(0.60, _yt2, f"AUC-MSE: {_fmt2(_auc)}",
                           fontname="Cambria", fontsize=10, color="#444444", ha="left", va="top")
                _fig6.text(0.82, _yt2, f"n={_npt}",
                           fontname="Cambria", fontsize=9,  color="#888888", ha="left", va="top")
                _yt2 -= 0.027
            _yt2 -= 0.006
            _yt2 = _hdiv_ent(_fig6, _yt2)

            _yt2 = _sec_title_ent(_fig6, _yt2, "Glosario")
            _glossary = [
                ("CV",        "Coeficiente de variación. Variabilidad relativa respecto a la media; valores altos indican menor consistencia entre pasos."),
                ("SampEn",    "Sample Entropy. Cuantifica la irregularidad del patrón de marcha. Muy bajo: rigidez; muy alto: desorganización."),
                ("AUC-MSE",   "Área bajo la curva de la Multiscale Entropy. Resume la complejidad de la marcha en múltiples escalas temporales."),
                ("n",         "Número de ciclos válidos usados para calcular la entropía."),
                ("Zancada",   "Ciclo completo de marcha: contacto de un talón hasta el siguiente contacto del mismo talón."),
                ("Velocidad", "Distancia recorrida por unidad de tiempo (m/s). Umbral clínico: 1.12 m/s (Studenski 2011)."),
                ("Cadencia",  "Número de pasos por minuto."),
            ]
            for _term, _defn in _glossary:
                if _yt2 < 0.06: break
                _glines = _wraplines(f"{_term}: {_defn}", 120)
                for _gi, _gline in enumerate(_glines):
                    if _yt2 < 0.06: break
                    _fig6.text(0.09, _yt2, _gline,
                               fontname="Cambria", fontsize=9,
                               color="#6A6161" if _gi == 0 else "#6A6161",
                               ha="left", va="top")
                    _yt2 -= 0.022

            footer_page_number(_fig6, page_no)
            pdf.savefig(_fig6, dpi=300)
            plt.close(_fig6)

def export_graphs_as_pngs(save_dir, fig_results, curves_r, curves_l):
    fig_results.savefig(os.path.join(save_dir, "1_parametros_espacio_temporales.png"),
                        dpi=450, bbox_inches='tight', facecolor="white")
    gaitResults = {"curves_r": curves_r, "curves_l": curves_l}
    _plot_lower_body_lumbar_to_file(gaitResults, os.path.join(save_dir, "2_angulos_lumbares_cuerpo_inferior.png"))
    _plot_pelvis_translations_to_file(gaitResults, os.path.join(save_dir, "3_traslaciones_pelvis.png"))
    _plot_arm_angles_to_file(gaitResults,          os.path.join(save_dir, "4_angulos_brazo.png"))

# =========================================================
# ===                    GUI                            ===
# =========================================================
def main_gui():
    root = tk.Tk()
    root.title("Análisis de Marcha")
    root.geometry("1000x720")

    style = ttk.Style(); style.theme_use("default")
    for item in ["TFrame","TNotebook","TNotebook.Tab","TLabel"]:
        style.configure(item, background="white")
    root.configure(bg="white")
    root.columnconfigure(0, weight=1); root.rowconfigure(1, weight=1)

    try:
        logo_img = load_logo_image(LOGO_PATH, max_width=LOGO_MAX_WIDTH)
        if logo_img:
            logo_lbl = ttk.Label(root, image=logo_img); logo_lbl.image = logo_img
            logo_lbl.grid(row=0, column=0, sticky="w", padx=12, pady=12)
    except Exception: pass

    nb = ttk.Notebook(root); nb.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
    tab_params  = ttk.Frame(nb)
    tab_results = ttk.Frame(nb)
    nb.add(tab_params,  text="Parámetros")
    nb.add(tab_results, text="Análisis Marcha")

    frm = ttk.Frame(tab_params, padding=12); frm.pack(fill="x")

    vars = {key: tk.StringVar() for key in ["nombre","edad","altura","trc","mot"]}
    vars.update({
        "sexo": tk.StringVar(value="Hombre"),
        "modo": tk.StringVar(value="Sobre suelo"),
        "fall_history_12m": tk.StringVar(),    
    })

    # ── Campo extra: Carpeta de sesión (para entropía multi-ensayo) ──
    vars["session_folder"] = tk.StringVar()

    labels_cfg = [
        (0, "Nombre:",          "nombre",  None),
        (1, "Edad (años):",     "edad",    None),
        (2, "Altura (m):",      "altura",  None),
    ]
    for row, lbl_txt, var_key, _ in labels_cfg:
        ttk.Label(frm, text=lbl_txt, background="white").grid(row=row, column=0, sticky="e", padx=6, pady=4)
        w = 40 if var_key == "nombre" else 12
        span = 2 if var_key == "nombre" else 1
        ttk.Entry(frm, textvariable=vars[var_key], width=w).grid(row=row, column=1, sticky="we",
                                                                   padx=6, pady=4, columnspan=span)

    ttk.Label(frm, text="Género:", background="white").grid(row=4, column=0, sticky="e", padx=6, pady=4)
    ttk.Combobox(frm, textvariable=vars["sexo"], values=["Hombre","Mujer"],
                 state="readonly", width=12).grid(row=4, column=1, sticky="w", padx=6, pady=4)

    ttk.Label(frm, text="Modo:", background="white").grid(row=5, column=0, sticky="e", padx=6, pady=4)
    ttk.Label(frm, text="Sobre suelo", background="white").grid(row=5, column=1, sticky="w", padx=6, pady=4)

    for i, ext in enumerate(["trc","mot"]):
        ttk.Label(frm, text=f"Archivo .{ext.upper()}:", background="white").grid(
            row=6+i, column=0, sticky="e", padx=6, pady=4)
        ttk.Entry(frm, textvariable=vars[ext], width=60).grid(
            row=6+i, column=1, sticky="we", padx=6, pady=4)
        cmd = (lambda v=vars[ext], e=ext:
               v.set(filedialog.askopenfilename(
                   title=f"Selecciona archivo .{e.upper()}",
                   filetypes=[(f"{e.upper()} files", f"*.{e}"),("Todos","*.*")]) or v.get()))
        ttk.Button(frm, text="Examinar…", command=cmd).grid(row=6+i, column=2, padx=6, pady=4)

    # ── Carpeta de sesión (opcional, para entropía) ──
    ttk.Label(frm, text="Carpeta sesión\n(entropía):", background="white",
              justify="right").grid(row=8, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(frm, textvariable=vars["session_folder"], width=60).grid(
        row=8, column=1, sticky="we", padx=6, pady=4)
    def _browse_session():
        p = filedialog.askdirectory(title="Seleccionar carpeta de sesión del paciente")
        if p: vars["session_folder"].set(p)
    ttk.Button(frm, text="Examinar…", command=_browse_session).grid(row=8, column=2, padx=6, pady=4)
    ttk.Label(frm, text="(Opcional: selecciona la carpeta con MarkerData/ y OpenSimData/ "
                        "para calcular la entropía con todos los ensayos)*",
              background="white", foreground="#666", font=("TkDefaultFont",8)).grid(
        row=9, column=1, columnspan=3, sticky="w", padx=6)
    ttk.Label(frm, text="También puedes dejarlo vacío y el análisis se hará solo con el ensayo actual (no recomendable). ",
              background="white", foreground="#666", font=("TkDefaultFont",8)).grid(
        row=10, column=1, columnspan=3, sticky="w", padx=6)
    ttk.Label(frm, text=" ",
              background="white", foreground="#666", font=("TkDefaultFont",8)).grid(
        row=11, column=1, columnspan=3, sticky="w", padx=6)
    ttk.Label(frm, text="*Al cargar la carpeta también puedes generar gráficos adicionales. ",
              background="white", foreground="#666", font=("TkDefaultFont",8)).grid(
        row=12, column=1, columnspan=3, sticky="w", padx=6)

    btn = ttk.Button(frm, text="Ejecutar análisis"); 
    btn.grid(row=14, column=0, columnspan=3, pady=10)
    frm.columnconfigure(1, weight=1)

    # Historial de caída 12 meses
    ttk.Label(frm, text="Caída últimos 12 meses:", background="white").grid(row=0, column=3, sticky="e", padx=6, pady=4)
    ttk.Combobox(frm, textvariable=vars["fall_history_12m"],
                values=["Sí","No"], state="readonly", width=5).grid(row=0, column=4, sticky="w", padx=6, pady=4)

    # ── Pestaña Resultados ──
    result_top = ttk.Frame(tab_results, padding=(10,10,10,4)); result_top.pack(fill="x")
    status_lbl = ttk.Label(result_top, text="Listo para ejecutar.", foreground="#444")
    status_lbl.pack(side="left")

    export_buttons_frame = ttk.Frame(result_top, style="TFrame")
    export_buttons_frame.pack(side="right")
    export_png_btn = ttk.Button(export_buttons_frame, text="Exportar PNG", state="disabled")
    export_png_btn.pack(side="right", padx=(5,0))
    export_btn = ttk.Button(export_buttons_frame, text="Exportar PDF", state="disabled")
    export_btn.pack(side="right", padx=(5,0))
    export_additional_btn = ttk.Button(export_buttons_frame, text="Gráficos Adicionales", state="disabled")
    export_additional_btn.pack(side="right", padx=(5,0))

    # ── Pestaña Entropía / Riesgo ──
    tab_entropy = ttk.Frame(nb)
    nb.add(tab_entropy, text="Riesgo Caída")

    entropytop = ttk.Frame(tab_entropy, style="TFrame")
    entropytop.pack(fill="x")
    export_entropy_btn = ttk.Button(entropytop, text="Exportar PDF Completo", state="disabled")
    export_entropy_btn.pack(side="right", padx=5, pady=0)

    # ── Banner de clasificación de riesgo ──
    risk_banner = tk.Frame(tab_entropy, bg="white", pady=6)
    risk_banner.pack(fill="x", padx=12, pady=(10, 0))

    # ── Fila 1: clínico (izquierda) + entrópico (derecha) ──
    risk_row1 = tk.Frame(risk_banner, bg="white")
    risk_row1.pack(fill="x")

    # Bloque clínico
    risk_clinical_frame = tk.Frame(risk_row1, bg="white")
    risk_clinical_frame.pack(side="left", padx=(8, 20))

    risk_level_lbl = tk.Label(risk_clinical_frame, text="", font=("Segoe UI", 13, "bold"),
                            bg="white", fg="#333333", anchor="w")
    risk_level_lbl.pack(side="left", padx=(0, 6))

    risk_score_lbl = tk.Label(risk_clinical_frame, text="", font=("Segoe UI", 10),
                            bg="white", fg="#555555", anchor="w")
    risk_score_lbl.pack(side="left")

    # Separador vertical
    tk.Frame(risk_row1, bg="#dddddd", width=1).pack(side="left", fill="y", pady=4)

    # Bloque entrópico
    risk_entropy_frame = tk.Frame(risk_row1, bg="white")
    risk_entropy_frame.pack(side="left", padx=(20, 8))

    risk_ent_level_lbl = tk.Label(risk_entropy_frame, text="", font=("Segoe UI", 13, "bold"),
                                bg="white", fg="#333333", anchor="w")
    risk_ent_level_lbl.pack(side="left", padx=(0, 6))

    risk_ent_score_lbl = tk.Label(risk_entropy_frame, text="", font=("Segoe UI", 10),
                                bg="white", fg="#555555", anchor="w")
    risk_ent_score_lbl.pack(side="left", padx=(0, 12))

    # Sub-scores SampEn y AUC-MSE apilados a la derecha del entrópico
    risk_subscores_frame = tk.Frame(risk_entropy_frame, bg="white")
    risk_subscores_frame.pack(side="left")

    risk_sampen_lbl = tk.Label(risk_subscores_frame, text="", font=("Segoe UI", 9),
                                bg="white", fg="#555555", anchor="w")
    risk_sampen_lbl.pack(anchor="w")

    risk_aucmse_lbl = tk.Label(risk_subscores_frame, text="", font=("Segoe UI", 9),
                                bg="white", fg="#555555", anchor="w")
    risk_aucmse_lbl.pack(anchor="w")

    # ── Separador horizontal ──
    tk.Frame(risk_banner, bg="#dddddd", height=1).pack(fill="x", padx=8, pady=(6, 4))

    # ── Fila 2: score combinado centrado ──
    risk_row2 = tk.Frame(risk_banner, bg="white")
    risk_row2.pack(fill="x")

    risk_comb_level_lbl = tk.Label(risk_row2, text="", font=("Segoe UI", 13, "bold"),
                                    bg="white", fg="#333333", anchor="center")
    risk_comb_level_lbl.pack(side="left", padx=(8, 6))

    risk_comb_score_lbl = tk.Label(risk_row2, text="", font=("Segoe UI", 10),
                                    bg="white", fg="#555555", anchor="w")
    risk_comb_score_lbl.pack(side="left")

    entropy_text = tk.Text(tab_entropy, wrap="word", font=("Segoe UI", 10),
                           bg="#f9f9f9", relief="flat", state="disabled",
                           padx=18, pady=10, spacing1=2, spacing3=3,
                           cursor="arrow")
    entropy_scroll = ttk.Scrollbar(tab_entropy, orient="vertical", command=entropy_text.yview)
    entropy_text.configure(yscrollcommand=entropy_scroll.set)

    entropy_text.tag_configure("section_hdr",  font=("Segoe UI", 10, "bold"), foreground="#2c3e50",
                                spacing1=10, spacing3=2, lmargin1=10)
    entropy_text.tag_configure("divider",      font=("Segoe UI", 8),  foreground="#cccccc",
                                spacing1=0,  spacing3=4, lmargin1=10)
    entropy_text.tag_configure("factor_item",  font=("Segoe UI", 10), foreground="#7b241c",
                                lmargin1=24, lmargin2=24)
    entropy_text.tag_configure("stat_item",    font=("Segoe UI", 10), foreground="#1a5276",
                                lmargin1=24, lmargin2=24)
    entropy_text.tag_configure("entropy_item", font=("Segoe UI", 9),  foreground="#444444",
                                lmargin1=24, lmargin2=24)
    entropy_text.tag_configure("interp_item",  font=("Segoe UI", 9, "italic"), foreground="#555555",
                                lmargin1=24, lmargin2=24, spacing3=2)
    entropy_text.tag_configure("score_line",   font=("Segoe UI", 10), foreground="#555555",
                                lmargin1=14, lmargin2=14)
    entropy_text.tag_configure("no_data",      font=("Segoe UI", 10, "italic"), foreground="#999999",
                                lmargin1=14)
    entropy_scroll.pack(side="right", fill="y")
    entropy_text.pack(fill="both", expand=True)

    result_main      = ttk.Frame(tab_results, padding=(10,0,10,10)); result_main.pack(fill="both", expand=True)
    canvas_container = ttk.Frame(result_main); canvas_container.pack(fill="both", expand=True)
    toolbar_container= ttk.Frame(result_main); toolbar_container.pack(fill="x")

    last_analysis = {}

    def set_busy(state=True): root.config(cursor="watch" if state else ""); root.update()

    def ejecutar():
        # --- Validar entradas básicas ----------------------------------------
        try:
            trc = vars["trc"].get().strip()
            mot = vars["mot"].get().strip()
            age = int(vars["edad"].get())
            hgt = float(vars["altura"].get().replace(",", "."))

            if not (18 <= age <= 120) or hgt <= 0 or not os.path.isfile(trc) or not os.path.isfile(mot):
                raise ValueError()
        except (ValueError, TypeError):
            messagebox.showerror(
                "Entradas inválidas",
                "Verifica todos los campos (edad, altura, archivos)."
            )
            return

        # --- Ejecutar análisis y entropía ------------------------------------
        try:
            status_lbl.config(text="Procesando…")
            set_busy(True)

            # 1) Análisis de marcha
            fig_final, info = run_full_analysis_offline_colors(
                trc, mot, age, vars["sexo"].get(), hgt, modo=vars["modo"].get()
            )

            # 2) Calcular riesgo de caída por entropía (si está disponible)
            fall_risk_result = None
            if _HAS_ENTROPY:
                session_folder = vars["session_folder"].get().strip()
                fh_raw = vars["fall_history_12m"].get().strip().lower()
                has_fall = fh_raw in ("sí", "si", "yes", "1", "true")

                try:
                    if session_folder and os.path.isdir(session_folder):
                        status_lbl.config(text="Calculando entropía multi‑ensayo…")
                        root.update()
                        fall_risk_result = analyze_fall_risk_from_session_folder(
                            session_folder, gait_analysis,
                            fall_history=has_fall,
                            height=hgt,
                            verbose=True
                        )
                    else:
                        fall_risk_result = analyze_fall_risk_from_gait_analysis(
                            info["scal_r_all"], info["scal_l_all"],
                            fall_history=has_fall,
                            height=hgt,
                            verbose=True
                        )
                except Exception as e_ent:
                    print("[Entropía] Error:", e_ent)
                    import traceback; traceback.print_exc()

            # 3) Mostrar figura en pestaña Resultados
            for w in canvas_container.winfo_children():
                w.destroy()
            for w in toolbar_container.winfo_children():
                w.destroy()

            canvas = FigureCanvasTkAgg(fig_final, master=canvas_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(canvas, toolbar_container).update()

            nb.select(tab_results)

            # 4) Panel de riesgo de caída en la GUI
            if fall_risk_result:
                fr = fall_risk_result.get("fall_risk", {})
                level = fr.get("risk_level", "N/D")
                entropy_note = fr.get("entropy_note", "")
                n_tr = fall_risk_result.get("n_trials", 1)
                source = (
                    "multi-ensayo"
                    if (vars["session_folder"].get().strip()
                        and os.path.isdir(vars["session_folder"].get().strip()))
                    else "ensayo único"
                )

                color_map = {"ALTO": "#c0392b", "MODERADO": "#e67e22", "BAJO": "#27ae60"}
                fg = color_map.get(str(level), "#333333")

                # ── Rellenar pestaña Entropía / Riesgo ──
                gs_  = fall_risk_result.get("gait_stats", {})
                ent_ = fall_risk_result.get("entropies", {})
                fr_  = fall_risk_result.get("fall_risk", {})
                fh_  = fr_.get("fall_history_applied", False)
                sb_  = fr_.get("score_base", "")
                sf_  = fr_.get("score_final", "")

                # ── Rellenar pestaña Entropía con interfaz visual ──
                # ── Actualizar banner visual ──
                _cmap = {"ALTO": "#c0392b", "MODERADO": "#e67e22", "BAJO": "#27ae60"}
                _bgc  = {"ALTO": "#fdf3f2", "MODERADO": "#fef9f2", "BAJO": "#f2faf5"}
                _lv   = str(level).upper()
                _color  = _cmap.get(_lv, "#333333")
                _bgcol  = _bgc.get(_lv, "white")
                risk_banner.config(bg=_bgcol)
                _sb   = fr_.get("score_base")
                _sf   = fr_.get("score_final")
                _s_se = fr_.get("score_sampen")
                _s_auc= fr_.get("score_aucmse")
                _fh   = fr_.get("fall_history_applied", False)
                _ntr     = fall_risk_result.get("ntrials", 1)
                _hist    = f" +15 historial → {_sf}" if _fh else ""

                def _fmt_es(v):
                    import math
                    if v is None: return "N/A"
                    try: return "N/A" if math.isnan(float(v)) else f"{float(v):.0f}"
                    except: return "N/A"

                # Calcular scores entropía y combinado
                _s_ent_total = None
                if _s_se is not None and _s_auc is not None:
                    try: _s_ent_total = float(_s_se) + float(_s_auc)
                    except (TypeError, ValueError): pass

                _s_combined = None
                if _sb is not None and _s_ent_total is not None:
                    try: _s_combined = 0.60 * float(_sb) + 0.40 * float(_s_ent_total)
                    except (TypeError, ValueError): pass

                def _ent_risk_level(score):
                    if score is None: return "N/D", "#333333"
                    if score >= 60:   return "ALTO",     "#c0392b"
                    if score >= 30:   return "MODERADO", "#e67e22"
                    return "BAJO", "#27ae60"

                _ent_level,  _ent_color  = _ent_risk_level(_s_ent_total)
                _comb_level, _comb_color = _ent_risk_level(_s_combined)

                # Actualizar banner
                risk_row1.config(bg=_bgcol)
                risk_row2.config(bg=_bgcol)
                risk_clinical_frame.config(bg=_bgcol)
                risk_entropy_frame.config(bg=_bgcol)
                risk_subscores_frame.config(bg=_bgcol)

                risk_level_lbl.config(text=f"⬤  Riesgo {_lv}", fg=_color, bg=_bgcol)
                risk_score_lbl.config(text=f"Score clínico: {_fmt_es(_sb)}/100", bg=_bgcol)

                risk_ent_level_lbl.config(text=f"⬤  Riesgo {_ent_level}", fg=_ent_color, bg=_bgcol)
                risk_ent_score_lbl.config(text=f"Score Entropía: {_fmt_es(_s_ent_total)}/100", bg=_bgcol)
                risk_sampen_lbl.config(text=f"Score SampEn:  {_fmt_es(_s_se)}/50", bg=_bgcol)
                risk_aucmse_lbl.config(text=f"Score AUC-MSE: {_fmt_es(_s_auc)}/50", bg=_bgcol)

                risk_comb_level_lbl.config(text=f"⬤  Riesgo Combinado (60/40): {_comb_level}", fg=_comb_color, bg=_bgcol)
                risk_comb_score_lbl.config(text=f"Score: {_s_combined:.1f}/100" if _s_combined is not None else "Score: N/D", bg=_bgcol)

                title_tag_ = ("title_alto"     if _lv == "ALTO"
                               else "title_moderado" if _lv == "MODERADO"
                               else "title_bajo")
                entropy_text.config(state="normal", bg="#f9f9f9")
                entropy_text.delete("1.0", "end")

                # ── Factores contribuyentes ──
                entropy_text.insert("end", "\n  Factores contribuyentes\n", "section_hdr")
                entropy_text.insert("end", "  " + "─" * 50 + "\n", "divider")
                factors_ = fr_.get("contributing_factors", [])
                if factors_:
                    for fi_ in factors_:
                        entropy_text.insert("end", f"  ▸  {fi_}\n", "factor_item")
                else:
                    entropy_text.insert("end", "  Sin factores de riesgo destacados.\n", "no_data")

                # ── Estadísticos de marcha ──
                entropy_text.insert("end", "\n  Estadísticos de marcha\n", "section_hdr")
                entropy_text.insert("end", "  " + "─" * 50 + "\n", "divider")
                stat_rows = [
                    ("stride_time_mean_s",    "Tiempo de zancada (media)  "),
                    ("stride_time_cv_pct",    "CV Tiempo de zancada "),
                    ("stride_length_mean_m",  "Longitud de zancada (media)  "),
                    ("stride_length_cv_pct",  "CV Longitud de zancada     "),
                    ("cadence_mean_spm",      "Cadencia (media)     "),
                    ("cadence_cv_pct",        "CV Cadencia          "),
                    ("walking_speed_mean_ms", "Velocidad Marcha (media) "),
                    (None,                    "   "), 
                ]
                # Mostrar en pares lado a lado (2 columnas)
                pairs = [(stat_rows[i], stat_rows[i+1] if i+1 < len(stat_rows) else None)
                         for i in range(0, len(stat_rows), 2)]
                for left_, right_ in pairs:
                    key_l, lab_l = left_
                    v_l = gs_.get(key_l)
                    _unit_map = {
                        "stride_time_mean_s":    " s",
                        "stride_time_cv_pct":    " %",
                        "stride_length_mean_m":  " m",
                        "stride_length_cv_pct":  " %",
                        "walking_speed_mean_ms": " m/s",
                        "cadence_mean_spm":      " pasos/min",
                        "cadence_cv_pct":        " %",
                    }
                    unit_l = _unit_map.get(key_l, "")
                    fmt_l  = ".1f" if key_l == "cadence_mean_spm" else ".3f"
                    left_txt = f"  {lab_l:<22}  {v_l:{fmt_l}}{unit_l}" if v_l is not None else ""
                    if right_:
                        key_r, lab_r = right_
                        v_r = gs_.get(key_r)
                        unit_r = _unit_map.get(key_r, "")
                        fmt_r  = ".1f" if key_r == "cadence_mean_spm" else ".3f"
                        right_txt = f"   {lab_r:<22}  {v_r:{fmt_r}}{unit_r}" if v_r is not None else ""
                    else:
                        right_txt = ""
                    if left_txt or right_txt:
                        entropy_text.insert("end", left_txt + right_txt + "\n", "stat_item")

                # ── Entropías ──
                entropy_text.insert("end", "\n  Entropías (descriptivas)\n", "section_hdr")
                entropy_text.insert("end", "  " + "─" * 50 + "\n", "divider")
                for ekey_, elab_ in [
                    ("stride_time", "Tiempo de zancada"),
                    ("stride_time_variability", "Variabilidad local del tiempo de zancada"),
                    ("trunk_ap_acceleration", "Aceleración AP del tronco"),
                ]:
                    ev_ = ent_.get(ekey_, {})
                    if ev_:
                        se_  = ev_.get("sampen",  "N/A")
                        mse_ = ev_.get("mse", {}).get("auc_mse", "N/A")
                        n_   = ev_.get("n",       "?")
                        se_s  = f"{se_:.3f}"  if isinstance(se_,  float) else str(se_)
                        mse_s = f"{mse_:.3f}" if isinstance(mse_, float) else str(mse_)
                        entropy_text.insert("end",
                            f"  {elab_:<22}  SampEn={se_s:<7}  AUC-MSE={mse_s:<7}  n={n_}\n",
                            "entropy_item")

                # ── Interpretación clínica generada ──
                interp_ = fall_risk_result.get("interpretation", "")
                entropy_text.insert("end", "\n  Interpretación Clínica\n", "section_hdr")
                entropy_text.insert("end", "  " + "─" * 50 + "\n", "divider")

                # Párrafo 1 — significado del nivel de riesgo
                _lv_es = str(level).upper()
                if _lv_es == "ALTO":
                    p1 = (
                        "  El perfil cinemático de este paciente presenta múltiples indicadores "
                        "de riesgo elevado de caída. Un score ≥60 puntos, especialmente con "
                        "historial de caídas previas (+15 puntos), refleja una combinación de "
                        "alteraciones en velocidad, variabilidad y dinámica de zancada que la "
                        "literatura asocia a un riesgo significativamente aumentado "
                        "(Hausdorff 2001, Studenski 2011, Tinetti 1988).\n"
                    )
                elif _lv_es == "MODERADO":
                    p1 = (
                        "  El perfil cinemático muestra señales de alerta en algunos parámetros "
                        "de marcha, sin alcanzar el umbral de riesgo alto. Un score en rango "
                        "moderado (30–59 puntos) indica que el paciente presenta alteraciones "
                        "parciales que pueden requerir seguimiento clínico y valoración "
                        "funcional periódica (Kressig 2004, Lord 2011).\n"
                    )
                else:
                    p1 = (
                        "  El perfil cinemático se encuentra dentro de los rangos normativos "
                        "para la mayoría de los parámetros evaluados. Un score bajo (<30 puntos) "
                        "indica una marcha estable con dinámica locomotora preservada. "
                        "Se recomienda mantener seguimiento preventivo anual "
                        "(Studenski 2011, Cesari 2005).\n"
                    )
                entropy_text.insert("end", p1, "interp_item")

                # Párrafo 2 — qué significan los factores detectados
                factors_ = fr_.get("contributing_factors", [])
                if factors_:
                    p2_lines = ["  Factores detectados y su significado clínico:\n"]
                    for fi_ in factors_:
                        fi_low = fi_.lower()
                        if "cv tiempo de zancada" in fi_low:
                            p2_lines.append(
                                "  · CV de tiempo de zancada elevado: mayor variabilidad en el tiempo "
                                "de zancada es el predictor más robusto de caídas en adultos mayores "
                                "(RR 1.007 por unidad, Hausdorff 2001). Refleja pérdida en la "
                                "regulación del ritmo locomotor.\n"
                            )
                        elif "velocidad" in fi_low:
                            p2_lines.append(
                                "  · Velocidad de marcha reducida: la velocidad es la variable "
                                "integradora más poderosa del estado funcional. Por debajo de "
                                "1.12 m/s se asocia a mayor mortalidad y riesgo de caída "
                                "(Studenski 2011, AUC 0.88).\n"
                            )
                        elif "cv longitud de zancada" in fi_low:
                            p2_lines.append(
                                "  · CV de longitud de zancada elevado: la variabilidad en longitud "
                                "de zancada (>4%) indica pérdida de consistencia en el patrón "
                                "espacial de la marcha (Hausdorff 2001).\n"
                            )
                        elif "longitud de zancada" in fi_low:
                            p2_lines.append(
                                "  · Longitud de zancada reducida: una longitud de zancada baja "
                                "respecto a la talla del paciente predice caídas con OR significativo "
                                "en modelos multivariantes (Oberg 1993, Bohannon 1997).\n"
                            )
                        elif "cv cadencia" in fi_low:
                            p2_lines.append(
                                "  · CV de cadencia elevado: alta variabilidad en cadencia (>5%) "
                                "se asocia a inestabilidad y mayor riesgo prospectivo de caídas "
                                "(Lord 2011).\n"
                            )
                    for line_ in p2_lines:
                        entropy_text.insert("end", line_, "interp_item")

                # Párrafo 3 — entropía
                ent_st_ = ent_.get("stride_time", {})
                if ent_st_:
                    se_val  = ent_st_.get("sampen")
                    mse_val = ent_st_.get("mse", {}).get("auc_mse")
                    p3_parts = []
                    if se_val is not None and se_val == se_val:
                        if se_val < 1.00 or se_val > 1.70:
                            p3_parts.append(
                                f"  · SampEn de stride time ({se_val:.3f}) fuera del rango normal "
                                "(1.20–1.50): patrón patológico, compatible con alteración marcada de la "
                                "complejidad locomotora (Stergiou 2011, Hausdorff 2001).\n"
                            )
                        elif se_val < 1.20 or se_val > 1.50:
                            p3_parts.append(
                                f"  · SampEn de stride time ({se_val:.3f}) en rango limítrofe "
                                "(1.00–1.20 o 1.50–1.70): señal de alerta en la complejidad del patrón "
                                "de marcha, requiere seguimiento (Stergiou 2011).\n"
                            )
                        else:
                            p3_parts.append(
                                f"  · SampEn de stride time ({se_val:.3f}) dentro del rango normal "
                                "(1.20–1.50): dinámica locomotora con complejidad preservada "
                                "(Hausdorff 2001, Stergiou 2011).\n"
                            )
                    if mse_val is not None and mse_val == mse_val:
                        if mse_val < 0.82 or mse_val > 1.28:
                            p3_parts.append(
                                f"  · AUC-MSE ({mse_val:.3f}) fuera del rango normal (0.92–1.15): "
                                "patrón patológico, pérdida de complejidad multiescala, predictor "
                                "independiente de inestabilidad funcional (Lipsitz 2004, Costa 2002).\n"
                            )
                        elif mse_val < 0.92 or mse_val > 1.15:
                            p3_parts.append(
                                f"  · AUC-MSE ({mse_val:.3f}) en rango limítrofe (0.82–0.92 o "
                                "1.15–1.28): señal de alerta en la complejidad multiescala de la marcha "
                                "(Costa 2002).\n"
                            )
                        else:
                            p3_parts.append(
                                f"  · AUC-MSE ({mse_val:.3f}) dentro del rango normal (0.92–1.15): "
                                "complejidad multiescala preservada (Costa 2002).\n"
                            )
                    if p3_parts:
                        entropy_text.insert("end", "  Análisis de complejidad de la marcha:\n", "interp_item")
                        for p_ in p3_parts:
                            entropy_text.insert("end", p_, "interp_item")

                # Pie de fuentes + interpretación combinada
                _combined_ = fr_.get("combined_interpretation", "")
                if _combined_:
                    entropy_text.insert("end", "\n Interpretación combinada\n", "section_hdr")
                    entropy_text.insert("end", "  " + "─" * 50 + "\n", "divider")
                    entropy_text.insert("end", f"  {_combined_}\n", "interp_item")
                else:
                    entropy_text.insert("end",
                        "\n  ─ Score acumulado basado en umbrales validados en literatura\n"
                        "  (Hausdorff 2001, Studenski 2011, Kressig 2004, Tinetti 1988, "
                        "Stergiou 2011, Costa 2002, Lipsitz 2004).\n",
                        "interp_item"
                    )
                if interp_:
                    entropy_text.insert("end", f"\n  {interp_}\n", "interp_item")

                entropy_text.insert("end", "\nGlosario\n", "section_hdr")
                entropy_text.insert("end", ("-" * 50) + "\n", "divider")

                glossary_lines = [
                    "CV: coeficiente de variación. Mide cuánta variabilidad hay respecto a la media; valores más altos indican menor consistencia entre pasos.",
                    "SampEn: Sample Entropy o entropía muestral. Cuantifica la irregularidad del patrón de marcha; valores muy bajos sugieren rigidez y valores muy altos, desorganización.",
                    "AUC-MSE: área bajo la curva de la Multiscale Entropy. Resume la complejidad de la marcha en múltiples escalas temporales.",
                    "Tiempo de zancada: cuánto tarda un ciclo completo de marcha.",
                    "Longitud de zancada: distancia recorrida en un ciclo completo.",
                    "Velocidad de la marcha: distancia recorrida por unidad de tiempo.",
                    "Cadencia: número de pasos por minuto.",
                    "n: cantidad de segmentos o datos válidos usados para calcular la entropía.",
                    "Riesgo clínico: puntuación basada en umbrales espacio-temporales clásicos de marcha.",
                    "Riesgo por entropía: estimación complementaria basada en complejidad e irregularidad de la dinámica de la marcha.",
                ]

                for gl in glossary_lines:
                    entropy_text.insert("end", f"• {gl}\n", "interp_item")

                entropy_text.config(state="disabled")
                nb.tab(tab_entropy, text=f"Entropía / Riesgo  [{level}]")

            else:

                risk_level_lbl.config(text="Sin datos de riesgo", fg="#999999", bg="white")
                risk_score_lbl.config(text="", bg="white")
                risk_banner.config(bg="white")
                risk_ent_level_lbl.config(text="", fg="#999999", bg="white")
                risk_ent_score_lbl.config(text="", bg="white")
                risk_comb_level_lbl.config(text="", fg="#999999", bg="white")
                risk_comb_score_lbl.config(text="", bg="white")
                entropy_text.config(state="normal", bg="#f9f9f9")
                entropy_text.delete("1.0", "end")
                entropy_text.insert("end", "No hay datos de entropía disponibles.", "no_data")
                entropy_text.config(state="disabled")
                nb.tab(tab_entropy, text="Entropía / Riesgo")

            # 5) Guardar último análisis para exportar / dataset
            last_analysis.clear()
            last_analysis.update({
                "fig":              fig_final,
                "info":             info,
                "patient":          {k: v.get() for k, v in vars.items()},
                "fall_risk_result": fall_risk_result,
            })

            export_btn.config(state="normal")
            export_png_btn.config(state="normal")
            export_additional_btn.config(state="normal")
            if fall_risk_result:
                export_entropy_btn.config(state="normal")
            status_lbl.config(text=f"Análisis completado para {os.path.basename(trc)}")

        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Error", f"Ocurrió un problema:\n{e}")
            status_lbl.config(text="Error en la ejecución.")
        finally:
            set_busy(False)

    btn.config(command=ejecutar)

    def do_export_pdf():
        if not last_analysis:
            messagebox.showwarning("Exportar PDF", "Primero ejecuta un análisis."); return
        patient_name = last_analysis["patient"].get('nombre') or 'paciente'
        trial_name   = last_analysis["info"].get('trial_name', '')
        default_name = f"Reporte_{patient_name.replace(' ','_')}_{trial_name}.pdf"
        save_path = filedialog.asksaveasfilename(title="Guardar reporte PDF",
                                                  defaultextension=".pdf", initialfile=default_name)
        if not save_path: return
        try:
            set_busy(True)
            export_pdf_report(
                save_path=save_path, logo_path=LOGO_PATH,
                patient=last_analysis["patient"], trial_name=trial_name,
                fig_results=last_analysis["fig"],
                metrics=last_analysis["info"].get("metrics", {}),
                curves_r=last_analysis["info"].get("curves_r"),
                curves_l=last_analysis["info"].get("curves_l"),
                fall_risk_result=None,
            )
            messagebox.showinfo("Exportar PDF", f"Reporte guardado en:\n{save_path}")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Exportar PDF", f"No se pudo crear el PDF:\n{e}")
        finally:
            set_busy(False)

    def do_export_pngs():
        if not last_analysis:
            messagebox.showwarning("Exportar PNG", "Primero ejecuta un análisis."); return
        base_dir = filedialog.askdirectory(title="Seleccionar carpeta para guardar los gráficos PNG")
        if not base_dir: return
        try:
            set_busy(True)
            patient_name = last_analysis["patient"].get('nombre') or 'paciente'
            trial_name   = last_analysis["info"].get('trial_name', '')
            folder_name  = f"Reporte_{patient_name.replace(' ','_')}_{trial_name}"
            save_dir     = os.path.join(base_dir, folder_name)
            os.makedirs(save_dir, exist_ok=True)
            export_graphs_as_pngs(save_dir=save_dir, fig_results=last_analysis["fig"],
                                   curves_r=last_analysis["info"].get("curves_r"),
                                   curves_l=last_analysis["info"].get("curves_l"))
            messagebox.showinfo("Exportar PNG", f"Gráficos guardados en:\n{save_dir}")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Exportar PNG", f"No se pudieron crear los archivos PNG:\n{e}")
        finally:
            set_busy(False)

    export_btn.config(command=do_export_pdf)
    export_png_btn.config(command=do_export_pngs)

    # ── Exportar PDF solo de Entropía / Riesgo ──────────────────────────────
    def do_export_entropy_pdf():
        if not last_analysis or not last_analysis.get("fall_risk_result"):
            messagebox.showwarning("Exportar PDF Entropía", "Primero ejecuta un análisis con entropía.")
            return

        patient_name = last_analysis["patient"].get("nombre") or "paciente"
        trial_name   = last_analysis["info"].get("trial_name", "ensayo")
        default_name = f"Reporte_Completo_{patient_name.replace(' ', '_')}_{trial_name}.pdf"
        save_path = filedialog.asksaveasfilename(
            title="Guardar PDF Completo con Entropía",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF", "*.pdf"), ("Todos", "*.*")]
        )
        if not save_path:
            return
        try:
            set_busy(True)
            export_pdf_report(
                save_path=save_path,
                logo_path=LOGO_PATH,
                patient=last_analysis["patient"],
                trial_name=trial_name,
                fig_results=last_analysis["fig"],
                metrics=last_analysis["info"].get("metrics", {}),
                curves_r=last_analysis["info"].get("curves_r"),
                curves_l=last_analysis["info"].get("curves_l"),
                fall_risk_result=last_analysis["fall_risk_result"],
            )
            messagebox.showinfo("Exportar PDF", f"PDF guardado en {save_path}")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Exportar PDF Entropía", f"Error al crear PDF:\n{e}")
        finally:
            set_busy(False)

    export_entropy_btn.config(command=do_export_entropy_pdf)

    # ── Exportar Gráficos Adicionales ──────────────────────────────
    def do_export_additional():
        if not last_analysis:
            messagebox.showwarning("Gráficos Adicionales", "Primero ejecuta un análisis.")
            return

        session = vars["session_folder"].get().strip()
        if not session or not os.path.isdir(session):
            messagebox.showwarning(
                "Gráficos Adicionales",
                "No es posible generar los gráficos adicionales.\nAgregue la carpeta de la sesión."
            )
            return

        output_dir = os.path.join(session, "Gráficos Adicionales")
        os.makedirs(output_dir, exist_ok=True)

        try:
            set_busy(True)
            status_lbl.config(text="Generando gráficos adicionales...")
            root.update()
            generate_additional_report(
                session_folder=session,
                output_dir=output_dir
            )
            messagebox.showinfo("Gráficos Adicionales", f"Gráficos guardados en:\n{output_dir}")
            status_lbl.config(text="Gráficos adicionales exportados correctamente.")
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Gráficos Adicionales", f"Error al generar gráficos:\n{e}")
            status_lbl.config(text="Error al generar gráficos adicionales.")
        finally:
            set_busy(False)

    export_additional_btn.config(command=do_export_additional)

    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), sys.exit(0)))
    root.mainloop()

if __name__ == "__main__": 
    main_gui()