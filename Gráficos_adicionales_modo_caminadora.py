"""
graficos_williams_v8.py
-----------------------
Genera TODOS los gráficos:

ESCALARES (scatter por ciclo):
  cadence_all_cycles.png, stride_length_all_cycles.png,
  step_width_all_cycles.png, gait_speed_all_cycles.png

SERIES TEMPORALES ciclo a ciclo (todas las variables):
  *_time_series.png

BOXPLOTS por ensayo (reproducibilidad):
  *_boxplot_by_trial.png

ANÁLISIS DE SEÑALES (stride time):
  stride_time_series.png, poincare_stride_time.png,
  acf_stride_time.png

Requiere: gait_analysissinopensim12.py y gait_analysis_offline24espanol_NEW8.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.spatial import KDTree
from scipy.stats import linregress

from gait_analysissinopensim12 import gait_analysis
from gait_analysis_offline_modo_caminadora4 import (
    load_trc_opensim,
    pdkv_by_stride_from_trc,
    compute_foot_clearance_from_trc,
)

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

SESSION_DIR = None
OUTPUT_DIR  = None

LEGS           = ["r", "l"]
FILTER_FREQ    = 6
TRIM_START     = 0.05
TRIM_END       = 0.05
N_GAIT_CYCLES  = -1
GAIT_STYLE     = "auto"
MAX_MSE_SCALE  = 6
EXCLUDE_TRIALS = []

SCALAR_MAP = {
    "double_support_time":    ("Doble apoyo", "%"),
    "step_length_symmetry":   ("Simetría de longitud de paso", "%"),
    "step_width":              ("Ancho de paso", "m"),
    "gait_speed":              ("Velocidad de marcha", "m/s"),
    "cadence":                 ("Cadencia", "pasos/min"),
}

COLOR = {"R": "#1a6faf", "L": "#c0392b"}

# ══════════════════════════════════════════════════════════════
# DETECCIÓN DE ENSAYOS
# ══════════════════════════════════════════════════════════════

def find_trials(session_dir):
    trc_files = glob.glob(os.path.join(session_dir, "**", "*.trc"), recursive=True)
    mot_files = glob.glob(os.path.join(session_dir, "**", "*.mot"), recursive=True)
    trc_map = {os.path.splitext(os.path.basename(f))[0]: f for f in trc_files}
    mot_map = {os.path.splitext(os.path.basename(f))[0]: f for f in mot_files}
    common = sorted(set(trc_map.keys()) & set(mot_map.keys()))
    trials = []
    for name in common:
        if any(ex in name for ex in EXCLUDE_TRIALS):
            continue
        trials.append({"trial_name": name, "trc": trc_map[name], "mot": mot_map[name]})
    return trials


def run_ga(session_dir, trial_name, leg):
    return gait_analysis(
        session_dir, trial_name, leg=leg,
        lowpass_cutoff_frequency_for_coordinate_values=FILTER_FREQ,
        n_gait_cycles=N_GAIT_CYCLES,
        gait_style=GAIT_STYLE,
        trimming_start=TRIM_START,
        trimming_end=TRIM_END,
    )

# ══════════════════════════════════════════════════════════════
# EXTRACCIÓN DE DATOS
# ══════════════════════════════════════════════════════════════
def extract_scalar_series(ga, name):
    try:
        scalars = ga.compute_scalars([name], return_all=True)
        if name not in scalars:
            return np.array([], dtype=float)
        v = np.asarray(scalars[name]["value"], dtype=float).ravel()
        return v[np.isfinite(v)]
    except Exception as e:
        print(f"  ! No se pudo extraer {name}: {e}")
        return np.array([], dtype=float)

def extract_stride_times(ga):
    t  = ga.gaitEvents["ipsilateralTime"]
    st = np.diff(t, axis=1)[:, 1].squeeze()
    st = np.asarray(st, dtype=float).ravel()
    return st[np.isfinite(st)]

def extract_valgus_series(trc_path, side):
    try:
        df_trc = load_trc_opensim(trc_path)
        result = pdkv_by_stride_from_trc(df_trc, side)
        vals = np.asarray(result["peaks_deg"], dtype=float)
        vals = vals[np.isfinite(vals)]
        start = 0 if side == "R" else 1
        return filter_outliers(vals[start::2])
    except Exception as e:
        print(f"  [!] Valgo dinámico ({side}): {e}")
        return np.array([], dtype=float)

def extract_foot_clearance_series(trc_path, side):
    try:
        df_trc = load_trc_opensim(trc_path)
        # Llamás con los marcadores del lado pedido (igual que antes)
        result = compute_foot_clearance_from_trc(
            df_trc, f"{side}BigToe", f"{side}Heel")
        vals = np.asarray(result["clearance_mm"], dtype=float)
        vals = vals[np.isfinite(vals)]
        # Filtrar: tomar solo 1 de cada 2 (pasos del mismo lado)
        start = 0 if side == "R" else 1
        return vals[start::2]
    except Exception as e:
        print(f"  [!] Foot clearance ({side}): {e}")
        return np.array([], dtype=float)

def filter_outliers(series: np.ndarray, n_sd: float = 3.0) -> np.ndarray:
    if series.size < 4:
        return series
    mean = np.mean(series)
    sd   = np.std(series, ddof=1)
    mask = np.abs(series - mean) <= n_sd * sd
    n_removed = series.size - mask.sum()
    if n_removed > 0:
        print(f"   [filtro] {n_removed} ciclo(s) eliminado(s) "
              f"(media={mean:.3f}, SD={sd:.3f})")
    return series[mask]

def extract_kinematic_sd_rom(curves_dict, side="r"):
    dfm = curves_dict["mean"]
    dfs = curves_dict["sd"]
    s = side.lower()

    targets_bilateral = [
        (f"knee_angle_{s}",    "flexión rodilla"),
        (f"ankle_angle_{s}",   "dorsiflexión tobillo"),
        (f"hip_flexion_{s}",   "flexión cadera"),
        (f"hip_adduction_{s}", "adducción cadera"),
        (f"hip_rotation_{s}",  "rotación cadera"),
    ]
    targets_shared = [
        ("pelvis_list",        "oblicuidad pélvica"),
        ("pelvis_tilt",        "inclinación pélvica"),
        ("pelvis_rotation",    "rotación pélvica"),
    ]

    result = {}
    for col, nice in targets_bilateral + targets_shared:
        if col in dfs.columns and col in dfm.columns:
            sd_mean = float(dfs[col].mean())
            rom     = float(dfm[col].max() - dfm[col].min())
            result[nice] = (sd_mean, rom)

    return result

def plot_kinematic_variability_per_trial(data_dict, trial_names,
                                          title, ylabel, output_path,
                                          color_r="#5e3c99", color_l="#b2ad00"):
    vars_available = [k for k in data_dict if
                      len(data_dict[k]["R"]) > 0 or len(data_dict[k]["L"]) > 0]
    if not vars_available:
        return

    n = len(vars_available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(12, 4 * nrows),
                              constrained_layout=True,
                              facecolor="white")
    axes = np.array(axes).flatten()

    x = np.arange(len(trial_names))
    short_names = [t.replace("Ensayo_", "E").replace("Prueba_", "P") 
                   for t in trial_names]

    for i, var in enumerate(vars_available):
        ax = axes[i]
        vals_r = data_dict[var]["R"]
        vals_l = data_dict[var]["L"]

        if vals_r:
            ax.plot(x[:len(vals_r)], vals_r, "o-", color=color_r,
                    lw=2, ms=6, label="Derecha")
        if vals_l:
            ax.plot(x[:len(vals_l)], vals_l, "s--", color=color_l,
                    lw=2, ms=6, label="Izquierda")

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(var, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.legend(fontsize=9, frameon=False)

    # Ocultar ejes sobrantes si n es impar
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] {output_path}")

# ══════════════════════════════════════════════════════════════
# ALGORITMOS DE SEÑAL
# ══════════════════════════════════════════════════════════════
def sample_entropy(x, m=2, r=None):
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 5:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x, ddof=1)
    if r == 0:
        return np.nan

    def _count(tlen):
        count = 0
        for i in range(N - tlen):
            t = x[i: i + tlen]
            for j in range(i + 1, N - tlen):
                if np.max(np.abs(t - x[j: j + tlen])) <= r:
                    count += 1
        return count

    B, A = _count(m), _count(m + 1)
    if B == 0:
        return np.nan
    if A == 0:
        return -np.log(1 / B)
    return -np.log(A / B)

def coarse_grain(x, scale):
    N = len(x)
    t = x[: N - (N % scale)] if N % scale != 0 else x
    return np.mean(t.reshape(-1, scale), axis=1)

def mse_curve(x, max_scale=MAX_MSE_SCALE, m=2):
    x       = np.asarray(x, dtype=float)
    eff_max = min(max_scale, max(1, len(x) // 4))
    scales, values = [], []
    for s in range(1, eff_max + 1):
        cg = coarse_grain(x, s)
        if len(cg) < 5:
            values.append(np.nan)
        else:
            r  = 0.2 * np.std(cg, ddof=1)
            se = sample_entropy(cg, m=m, r=r) if len(cg) >= 4 else np.nan
            values.append(se)
        scales.append(s)
    valid_idx = [(s, v) for s, v in zip(scales, values) if not np.isnan(v)]
    if len(valid_idx) >= 2:
        xs, ys = zip(*valid_idx)
        auc = float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))
    else:
        auc = float(valid_idx[0][1]) if len(valid_idx) == 1 else np.nan
    return {"scales": scales, "sampen": values, "auc": auc}

def autocorrelation(x, max_lag=20):
    x   = np.asarray(x, dtype=float)
    x   = x - np.mean(x)
    var = np.var(x)
    if var == 0 or len(x) < 4:
        return {"lags": [], "acf": []}
    acf = [float(np.mean(x[: len(x) - k] * x[k:]) / var)
           for k in range(max_lag + 1)]
    return {"lags": list(range(max_lag + 1)), "acf": acf}

# ══════════════════════════════════════════════════════════════
# GUARDADO
# ══════════════════════════════════════════════════════════════
def _legend_bottom(ax, ncol=4):
    """Mueve la leyenda debajo del eje X."""
    ax.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
    )

def _save(fig, output_dir, name, dpi=150, subfolder=None):
    folder = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] {path}")

# ══════════════════════════════════════════════════════════════
# GRÁFICO 1: ESCALARES (scatter por ciclo)
# ══════════════════════════════════════════════════════════════
def plot_scalar_all_trials(var_name, values_R, values_L, output_dir):
    if values_R.size == 0 and values_L.size == 0:
        print(f"   [!] Sin datos para '{var_name}'.")
        return
    label, units = SCALAR_MAP[var_name]

    COLOR_POINTS = {"R": "#5b9bd5", "L": "#c0392b"}
    COLOR_ERROR  = {"R": "#1a6faf", "L": "#922b21"}

    fig, ax = plt.subplots(figsize=(7, 5))
    jitter  = 0.06

    if values_R.size > 0:
        ax.scatter(np.random.uniform(-jitter, jitter, values_R.size),
                   values_R, alpha=0.6, color=COLOR_POINTS["R"],
                   edgecolor="none", label="Ciclos Derecha")
    if values_L.size > 0:
        ax.scatter(1 + np.random.uniform(-jitter, jitter, values_L.size),
                   values_L, alpha=0.6, color=COLOR_POINTS["L"],
                   edgecolor="none", label="Ciclos Izquierda")

    for cx, vals, side, pfx in [(0, values_R, "R", "Derecha"),
                                  (1, values_L, "L", "Izquierda")]:
        if vals.size > 0:
            m = vals.mean()
            s = vals.std(ddof=1) if vals.size > 1 else 0.0
            ax.errorbar(cx, m, yerr=s, fmt="o",
                        color=COLOR_ERROR[side], capsize=5, lw=2,
                        label=f"Media {pfx.lower()}: {m:.1f} ± {s:.1f} {units}")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ciclos Derecha", "Ciclos Izquierda"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(f"{label} [{units}]", fontsize=11)
    ax.set_title(label, fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _legend_bottom(ax)
    # Escalares (scatter)
    _save(fig, output_dir, f"{var_name}_todos_los_ciclos", subfolder="01_Escalares")

# ══════════════════════════════════════════════════════════════
# GRÁFICO 2: SERIE TEMPORAL CICLO A CICLO
# ══════════════════════════════════════════════════════════════
def plot_scalar_per_trial_series(per_trial_R, per_trial_L, trial_names,
                                  label, units, var_name, output_dir):

    TRIAL_COLORS = [
        "#1a6faf", "#e67e22", "#2ecc71", "#9b59b6",
        "#e74c3c", "#1abc9c", "#f39c12", "#34495e",
    ]

    has_R = any(v.size > 0 for v in per_trial_R)
    has_L = any(v.size > 0 for v in per_trial_L)
    if not has_R and not has_L:
        return

    panels = []
    if has_R:
        panels.append((per_trial_R, "Derecha"))
    if has_L:
        panels.append((per_trial_L, "Izquierda"))

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4.5 * n_panels), sharex=False)
    if n_panels == 1:
        axes = [axes]

    legend_handles = []

    for idx, (ax, (per_trial, side_label)) in enumerate(zip(axes, panels)):
        all_vals = []
        trial_lines = []

        for i, vals in enumerate(per_trial):
            if vals.size == 0:
                continue
            color = TRIAL_COLORS[i % len(TRIAL_COLORS)]
            x = np.arange(1, len(vals) + 1)
            short_name = trial_names[i] if i < len(trial_names) else f"E{i+1}"
            line, = ax.plot(x, vals, "o-", color=color, lw=1.5, ms=5,
                            alpha=0.85)
            trial_lines.append((line, f"E{i+1} ({short_name})"))
            all_vals.extend(vals.tolist())

        if all_vals:
            global_mean = np.mean(all_vals)
            mean_line = ax.axhline(global_mean, color="gray", ls="--", lw=1.2)
            # Media siempre en upper right, independiente de la leyenda de ensayos
            mean_legend = ax.legend(
                handles=[mean_line],
                labels=[f"Media = {global_mean:.3f} {units}"],
                fontsize=8, frameon=False,
                loc="upper right"
            )
            ax.add_artist(mean_legend)  # ← preserva esta leyenda cuando se añade la otra

        ax.set_xlabel("Ciclo #", fontsize=10)
        ax.set_ylabel(f"{label} [{units}]", fontsize=10)
        ax.set_title(f"{label} — {side_label}", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == n_panels - 1:
            legend_handles = trial_lines

    # Leyenda de ensayos solo en el panel de abajo
    if legend_handles:
        handles, labels = zip(*legend_handles)
        n_cols = min(len(handles), 4)
        axes[-1].legend(
            handles=list(handles),
            labels=list(labels),
            fontsize=8, frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=n_cols,
        )

    fig.suptitle(f"{label} — Serie ciclo a ciclo por ensayo",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, hspace=0.35)
    _save(fig, output_dir, f"{var_name}_series_por_ensayo",
          subfolder="02_Series_Temporales")

def plot_variable_time_series(series_R, series_L, label, units,
                               file_name, output_dir,
                               raw_R=None, raw_L=None):
    panels = [(series_R, raw_R, "R"), (series_L, raw_L, "L")]
    panels = [(s, r, side) for s, r, side in panels if s.size > 0]
    if not panels:
        print(f"   [!] Sin datos para '{label}', serie temporal omitida.")
        return

    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(10, 4 * len(panels)), sharex=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (series, raw, side) in zip(axes, panels):
        side_label = "Derecha" if side == "R" else "Izquierda"
        if raw is not None and raw.size > series.size:
            ax.plot(np.arange(1, len(raw) + 1), raw, "o-",
                    color="lightgray", lw=1, ms=3, alpha=0.6, label="Datos crudos")
            n_removed = raw.size - series.size
            ax.annotate(f"{n_removed} ciclo(s) eliminado(s) por filtro ±3 SD",
                        xy=(0.01, 0.04), xycoords="axes fraction",
                        fontsize=8, color="gray", style="italic")
        ax.plot(np.arange(1, len(series) + 1), series, "o-",
                color=COLOR[side], lw=1.5, ms=4, alpha=0.85, label="Datos filtrados")
        mean_val = np.mean(series)
        ax.axhline(mean_val, color="gray", ls="--", lw=1,
                   label=f"Media = {mean_val:.3f} {units}")
        ax.set_ylabel(f"{label} [{units}]", fontsize=10)
        ax.set_title(f"Serie de {label} — {side_label}", fontsize=11)
        ax.set_xlabel("Ciclo #", fontsize=10)
        _legend_bottom(ax)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    _save(fig, output_dir, file_name, subfolder="02_Series_Temporales")

# ══════════════════════════════════════════════════════════════
# GRÁFICO 3: BOXPLOT POR ENSAYO
# ══════════════════════════════════════════════════════════════
def plot_boxplot_by_trial(per_trial_R, per_trial_L, trial_names,
                           label, units, file_name, output_dir):
    if len(trial_names) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(f"{label} — Reproducibilidad por ensayo",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax, per_trial, side, base_color in zip(
            axes, [per_trial_R, per_trial_L], ["R", "L"],
            [COLOR["R"], COLOR["L"]]):
        side_label  = "Derecha" if side == "R" else "Izquierda"
        data_valid  = [v for v in per_trial if v.size > 0]
        labels_plot = [f"E{i+1}" for i, v in enumerate(per_trial)
                       if v.size > 0 and i < len(trial_names)]
        if not data_valid:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data_valid, patch_artist=True, notch=False, widths=0.5,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=4,
                                        markerfacecolor=base_color,
                                        alpha=0.5, linestyle="none"))
        for i, patch in enumerate(bp["boxes"]):
            alpha = 0.4 + 0.5 * (i / max(len(bp["boxes"]) - 1, 1))
            patch.set_facecolor(base_color)
            patch.set_alpha(alpha)

        all_vals    = np.concatenate(data_valid)
        global_mean = np.mean(all_vals)
        ax.axhline(global_mean, color="gray", ls="--", lw=1.2,
                   label=f"Media global = {global_mean:.3f} {units}")

        cv_texts = []
        for i, vals in enumerate(data_valid, start=1):
            if vals.size > 1:
                cv   = np.std(vals, ddof=1) / np.mean(vals) * 100
                ymax = np.percentile(vals, 75) + 1.5 * (
                    np.percentile(vals, 75) - np.percentile(vals, 25))
                ymax = max(ymax, np.max(vals))
                cv_texts.append((i, ymax, cv))

        data_min = np.min(all_vals)
        data_max = max(np.max(all_vals), max((t[1] for t in cv_texts), default=np.max(all_vals)))
        pad = (data_max - data_min) * 0.15
        ax.set_ylim(data_min - pad, data_max + pad * 1.8)

        for i, ymax, cv in cv_texts:
            ax.text(i, ymax + pad * 0.3, f"CV={cv:.1f}%",
                    ha="center", va="bottom", fontsize=7.5, color="dimgray",
                    clip_on=True)

        ax.set_xticks(range(1, len(labels_plot) + 1))
        ax.set_xticklabels(labels_plot, fontsize=9)
        ax.set_xlabel("Ensayo", fontsize=10)
        ax.set_ylabel(f"{label} [{units}]", fontsize=10)
        ax.set_title(side_label, fontsize=11, pad=10)
        ax.legend(fontsize=8, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.15), ncol=1)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.08)
    _save(fig, output_dir, file_name, subfolder="03_Boxplots_por_Ensayo")

# ══════════════════════════════════════════════════════════════
# GRÁFICOS DE SEÑAL (stride time)
# ══════════════════════════════════════════════════════════════
def plot_stride_time_series(series_R, series_L, output_dir,
                             raw_R=None, raw_L=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    for ax, series, raw, side in zip(axes,
                                      [series_R, series_L],
                                      [raw_R, raw_L],
                                      ["R", "L"]):
        if series.size == 0:
            ax.set_visible(False)
            continue
        if raw is not None and raw.size > series.size:
            ax.plot(np.arange(1, len(raw) + 1), raw, "o-",
                    color="lightgray", lw=1, ms=3, alpha=0.6, label="Datos crudos")
            ax.annotate(f"{raw.size - series.size} ciclo(s) eliminado(s) por filtro ±3 SD",
                        xy=(0.01, 0.04), xycoords="axes fraction",
                        fontsize=8, color="gray", style="italic")
        ax.plot(np.arange(1, len(series) + 1), series, "o-",
                color=COLOR[side], lw=1.5, ms=4, alpha=0.85, label="Datos filtrados")
        ax.axhline(np.mean(series), color="gray", ls="--", lw=1,
                   label=f"Media = {np.mean(series):.3f} s")
        ax.set_ylabel("Tiempo de zancada [s]", fontsize=10)
        ax.set_title(
            f"Serie de tiempos de zancada — {'Derecha' if side=='R' else 'Izquierda'}",
            fontsize=11)
        ax.set_xlabel("Ciclo #", fontsize=10)
        _legend_bottom(ax)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir, "series_tiempo_de_zancada", subfolder="04_Análisis_Señales")


def plot_poincare(series_R, series_L, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, series, side in zip(axes, [series_R, series_L], ["R", "L"]):
        if series.size < 4:
            ax.set_visible(False)
            continue
        x, y     = series[:-1], series[1:]
        diff     = (y - x) / np.sqrt(2)
        sumv     = (y + x) / np.sqrt(2)
        sd1      = np.std(diff, ddof=1)
        sd2      = np.std(sumv, ddof=1)
        ax.scatter(x, y, color=COLOR[side], alpha=0.6, s=25, edgecolor="none")
        theta    = np.linspace(0, 2 * np.pi, 200)
        cx_, cy_ = np.mean(x), np.mean(y)
        cos_a, sin_a = np.cos(np.pi / 4), np.sin(np.pi / 4)
        ex = cx_ + sd2 * np.cos(theta) * cos_a - sd1 * np.sin(theta) * sin_a
        ey = cy_ + sd2 * np.cos(theta) * sin_a + sd1 * np.sin(theta) * cos_a
        ax.plot(ex, ey, "k--", lw=1.2, alpha=0.7)
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], "gray", lw=0.8, ls=":")
        ax.set_xlabel("ST$_n$ [s]", fontsize=10)
        ax.set_ylabel("ST$_{n+1}$ [s]", fontsize=10)
        ax.set_title(
            f"Poincaré — {'Derecha' if side=='R' else 'Izquierda'}\n"
            f"SD1={sd1:.4f} s  SD2={sd2:.4f} s", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle("Diagrama de Poincaré — Tiempo de zancada",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir, "poincare_tiempo_de_zancada", subfolder="04_Análisis_Señales")

def plot_acf_fig(series_R, series_L, output_dir, max_lag=20):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, series, side in zip(axes, [series_R, series_L], ["R", "L"]):
        if series.size < 4:
            ax.set_visible(False)
            continue
        res      = autocorrelation(series, max_lag=max_lag)
        lags     = res["lags"]
        acf_vals = res["acf"]
        ci       = 1.96 / np.sqrt(len(series))
        ax.bar(lags, acf_vals, color=COLOR[side], alpha=0.7, width=0.6)
        ax.axhline(ci,  color="gray", ls="--", lw=1, label="IC 95%")
        ax.axhline(-ci, color="gray", ls="--", lw=1)
        ax.axhline(0,   color="black", lw=0.8)
        ax.set_xlim(-0.5, max_lag + 0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Lag (ciclos)", fontsize=10)
        ax.set_ylabel("ACF", fontsize=10)
        ax.set_title(
            f"Autocorrelación — {'Derecha' if side=='R' else 'Izquierda'}",
            fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Autocorrelación de Tiempo de Zancada",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir, "acf_tiempo_de_zancada",subfolder="04_Análisis_Señales")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    global SESSIONDIR, OUTPUTDIR
    trials = find_trials(SESSION_DIR)
    if not trials:
        print(f"No se encontraron ensayos en: {SESSION_DIR}")
        return

    trial_names = [t["trial_name"] for t in trials]
    print("Ensayos encontrados:", trial_names)

    scalar_per_trial = {var: {"R": [], "L": []} for var in SCALAR_MAP}
    valgus_per_trial = {"R": [], "L": []}
    fc_per_trial     = {"R": [], "L": []}
    stride_time_data = {"R": [], "L": []}

    kin_sd_per_trial  = {}
    kin_rom_per_trial = {}

    for trial in trials:
        tname = trial["trial_name"]
        trc_path = trial["trc"]
        print(f"\n{tname}")
        for leg in LEGS:
            side = "R" if leg == "r" else "L"
            try:
                ga = run_ga(SESSION_DIR, tname, leg)

                for var in SCALAR_MAP:
                    vals = extract_scalar_series(ga, var)
                    scalar_per_trial[var][side].append(
                        filter_outliers(vals) if vals.size > 0
                        else np.array([], dtype=float))
                    if vals.size > 0:
                        print(f"  {side} | {var}: {vals.size} ciclos")

                st = extract_stride_times(ga)
                if st.size > 0:
                    stride_time_data[side].append(st)
                    print(f"  {side} | series temporales: {st.size} ciclos")

                # ← BLOQUE NUEVO — con prints visibles
                #print(f"  {side} | intentando extraer cinemática...")
                curves_obj = ga.get_coordinates_normalized_time()
                #print(f"  {side} | curves_obj keys: {list(curves_obj.keys()) if isinstance(curves_obj, dict) else type(curves_obj)}")
                extracted = extract_kinematic_sd_rom(curves_obj, side=leg)
                #print(f"  {side} | extraído: {list(extracted.keys())}")
                for nice, (sd_val, rom_val) in extracted.items():
                    if nice not in kin_sd_per_trial:
                        kin_sd_per_trial[nice]  = {"R": [], "L": []}
                        kin_rom_per_trial[nice] = {"R": [], "L": []}
                    kin_sd_per_trial[nice][side].append(sd_val)
                    kin_rom_per_trial[nice][side].append(rom_val)

            except Exception as e:
                print(f"  [!] Error {tname} ({leg}): {e}")
                for var in SCALAR_MAP:
                    scalar_per_trial[var][side].append(np.array([], dtype=float))

        for side in ["R", "L"]:
            valgus = extract_valgus_series(trc_path, side)
            valgus_per_trial[side].append(
                filter_outliers(valgus) if valgus.size > 0
                else np.array([], dtype=float))
            fc = extract_foot_clearance_series(trc_path, side)
            fc_per_trial[side].append(
                filter_outliers(fc) if fc.size > 0
                else np.array([], dtype=float))

    def concat_all(per_trial_dict):
        return {
            side: np.concatenate(per_trial_dict[side])
            if any(v.size > 0 for v in per_trial_dict[side])
            else np.array([], dtype=float)
            for side in ["R", "L"]
        }

    scalar_all = {var: concat_all(scalar_per_trial[var]) for var in SCALAR_MAP}
    valgus_all = concat_all(valgus_per_trial)
    fc_all     = concat_all(fc_per_trial)

    series_R = (np.concatenate(stride_time_data["R"])
                if stride_time_data["R"] else np.array([], dtype=float))
    series_L = (np.concatenate(stride_time_data["L"])
                if stride_time_data["L"] else np.array([], dtype=float))
    raw_R, raw_L = series_R.copy(), series_L.copy()
    series_R = filter_outliers(series_R)
    series_L = filter_outliers(series_L)

    print(f"\nGenerando gráficos en: {OUTPUT_DIR}\n")

    # ── 1. Escalares (scatter) ────────────────────────────────
    print("── Escalares espacio-temporales ──")
    for var in SCALAR_MAP:
        plot_scalar_all_trials(
            var, scalar_all[var]["R"], scalar_all[var]["L"], OUTPUT_DIR)

    # ── 2. Series temporales ──────────────────────────────────
    print("\n── Series temporales ciclo a ciclo ──")
    for var, (label, units) in SCALAR_MAP.items():
        plot_variable_time_series(
            series_R=scalar_all[var]["R"], series_L=scalar_all[var]["L"],
            label=label, units=units,
            file_name=f"{var}_series_temporales", output_dir=OUTPUT_DIR)
    plot_variable_time_series(
        series_R=valgus_all["R"], series_L=valgus_all["L"],
        label="Valgo dinámico", units="°",
        file_name="valgo_dinamico_series_temporales", output_dir=OUTPUT_DIR)
    plot_variable_time_series(
        series_R=fc_all["R"], series_L=fc_all["L"],
        label="Amplitud del pie", units="mm",
        file_name="foot_clearance_series_temporales", output_dir=OUTPUT_DIR)
    
    # ── 2b. Series por ensayo (ciclo a ciclo) ─────────────────
    print("\n── Escalares por ensayo (ciclo a ciclo) ──")
    for var, (label, units) in SCALAR_MAP.items():
        plot_scalar_per_trial_series(
            per_trial_R=scalar_per_trial[var]["R"],
            per_trial_L=scalar_per_trial[var]["L"],
            trial_names=trial_names,
            label=label, units=units,
            var_name=var, output_dir=OUTPUT_DIR)

    #print("fc_per_trial R lengths:", [len(v) for v in fc_per_trial["R"]])
    #print("fc_per_trial L lengths:", [len(v) for v in fc_per_trial["L"]])
    #print("Total ensayos:", len(trial_names))

    plot_scalar_per_trial_series(
        per_trial_R=valgus_per_trial["R"],
        per_trial_L=valgus_per_trial["L"],
        trial_names=trial_names,
        label="Valgo dinámico", units="°",
        var_name="dynamic_valgus", output_dir=OUTPUT_DIR)

    plot_scalar_per_trial_series(
        per_trial_R=fc_per_trial["R"],
        per_trial_L=fc_per_trial["L"],
        trial_names=trial_names,
        label="Amplitud del pie", units="mm",
        var_name="foot_clearance", output_dir=OUTPUT_DIR)

    plot_scalar_per_trial_series(
        per_trial_R=stride_time_data["R"],
        per_trial_L=stride_time_data["L"],
        trial_names=trial_names,
        label="Tiempo de zancada", units="s",
        var_name="stride_time", output_dir=OUTPUT_DIR)

    # ── 3. Boxplots por ensayo ────────────────────────────────
    print("\n── Boxplots por ensayo (reproducibilidad) ──")
    for var, (label, units) in SCALAR_MAP.items():
        plot_boxplot_by_trial(
            per_trial_R=scalar_per_trial[var]["R"],
            per_trial_L=scalar_per_trial[var]["L"],
            trial_names=trial_names, label=label, units=units,
            file_name=f"{var}_boxplot_por_ensayo", output_dir=OUTPUT_DIR)
    plot_boxplot_by_trial(
        per_trial_R=valgus_per_trial["R"], per_trial_L=valgus_per_trial["L"],
        trial_names=trial_names, label="Valgo dinámico", units="°",
        file_name="valgo_dinamico_boxplot_por_ensayo", output_dir=OUTPUT_DIR)
    plot_boxplot_by_trial(
        per_trial_R=fc_per_trial["R"], per_trial_L=fc_per_trial["L"],
        trial_names=trial_names, label="Amplitud del pie", units="mm",
        file_name="foot_clearance_boxplot_por_ensayo", output_dir=OUTPUT_DIR)

    # ── 4. Análisis de señales ────────────────────────────────
    print("\n── Análisis de señales (tiempo de zancada) ──")
    plot_stride_time_series(series_R, series_L, OUTPUT_DIR,
                             raw_R=raw_R, raw_L=raw_L)
    plot_poincare(series_R, series_L, OUTPUT_DIR)
    plot_acf_fig(series_R, series_L, OUTPUT_DIR)

    print("\n── Variabilidad cinemática por ensayo ──")
    #print("DEBUG kin_sd_per_trial:", kin_sd_per_trial)
    plot_kinematic_variability_per_trial(
        kin_sd_per_trial, trial_names,
        title="Variabilidad cinemática por ensayo (SD media del ciclo)",
        ylabel="SD media (°)",
        output_path=os.path.join(OUTPUT_DIR, "Variabilidad_cinemática_por_ensayo.png")
    )
    plot_kinematic_variability_per_trial(
        kin_rom_per_trial, trial_names,
        title="Excursión articular por ensayo (ROM)",
        ylabel="ROM (°)",
        output_path=os.path.join(OUTPUT_DIR, "Excursión_articular_por_ensayo.png")
    )

def generate_additional_report(session_folder, output_dir):
    global SESSION_DIR, OUTPUT_DIR
    SESSION_DIR = session_folder
    OUTPUT_DIR = output_dir
    main()

if __name__ == "__main__":
    main()