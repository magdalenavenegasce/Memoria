#import os
#import utilsKinematicssinopensim
#from utils import download_kinematics
#from utilsPlotting import plot_dataframe
import numpy as np
import matplotlib.pyplot as plt

def color_bar_stride(ax, value_right, value_left, altura_paciente, label, fs=None):
    bar_height = 0.1
    fs_val = fs if fs else 9
    fs_title = fs if fs else 12
    fs_pad = -8 if fs else -15
    cutoff = 0.45 * altura_paciente
    max_val = max(cutoff*1.5, value_right*1.2, value_left*1.2)

    ax.barh(0, cutoff*0.8, left=0, color="red", height=bar_height)
    ax.barh(0, cutoff*0.2, left=cutoff*0.8, color="gold", height=bar_height)
    ax.barh(0, max_val-cutoff, left=cutoff, color="lime", height=bar_height)

    ax.vlines(value_right, ymin=-bar_height/2, ymax=bar_height/2, color="blue", linewidth=1.5, label="R")
    ax.text(value_right, bar_height/2, f"{value_right:.2f}", ha="center", va="bottom",
            fontsize=fs_val, color="blue", fontweight="bold")

    ax.vlines(value_left, ymin=-bar_height/2, ymax=bar_height/2, color="dimgrey", linewidth=1.5, label="L")
    ax.text(value_left, bar_height/2 + 0.06, f"{value_left:.2f}", ha="center", va="bottom",
            fontsize=fs_val, color="dimgrey", fontweight="bold")

    ax.set_xlim(0, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_title(label, fontsize=fs_title, pad=fs_pad)


def color_bar_stepwidth(ax, value, label, altura_paciente=None,
                        frac_low=0.043, frac_high=0.074,
                        yellow_margin_frac=0.008, mean=None, sd=None, fs=None):
    """
    Colorbar para step width basada en la altura del paciente.
    Verde:    4.3%–7.4% de la altura del paciente (en cm)
    Amarillo: margen de ±0.8% alrededor de la zona verde
    Rojo:     fuera de la zona amarilla
    """
    bar_height = 0.1
    fs_val = fs if fs else 9
    fs_title = fs if fs else 12
    fs_pad = -8 if fs else -15
    h = float(altura_paciente) * 100.0 if altura_paciente else 170.0  # cm
    good_low      = frac_low  * h
    good_high     = frac_high * h
    yellow_margin = yellow_margin_frac * h
    yellow_low    = good_low  - yellow_margin
    yellow_high   = good_high + yellow_margin
    min_val = max(0.0, yellow_low  - yellow_margin * 2)
    max_val = max(yellow_high + yellow_margin * 2, float(value) * 1.15)

    ax.barh(0, yellow_low  - min_val,      left=min_val,     color="red",  height=bar_height)
    ax.barh(0, good_low    - yellow_low,   left=yellow_low,  color="gold", height=bar_height)
    ax.barh(0, good_high   - good_low,     left=good_low,    color="lime", height=bar_height)
    ax.barh(0, yellow_high - good_high,    left=good_high,   color="gold", height=bar_height)
    ax.barh(0, max_val     - yellow_high,  left=yellow_high, color="red",  height=bar_height)

    ax.vlines(value, ymin=-bar_height/2, ymax=bar_height/2, color="black", linewidth=1.5)
    ax.text(value, bar_height/2 + 0.05, f"{value:.2f}", ha="center", va="bottom",
            fontsize=fs_val, color="black", fontweight="bold")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_xticks([round(min_val,1), round(yellow_low,1), round(good_low,1),
                   round(good_high,1), round(yellow_high,1), round(max_val,1)])
    ax.set_title(label, fontsize=fs_title, pad=fs_pad)

def color_bar_cadence(ax, value, label, fs=None):
    bar_height = 0.1
    fs_val = fs if fs else 9
    fs_title = fs if fs else 12
    fs_pad = -8 if fs else -15
    max_val = max(150, value*1.2)

    ax.barh(0, 70, left=20, color="red", height=bar_height)
    ax.barh(0, 10, left=90, color="gold", height=bar_height)
    ax.barh(0, max_val-100, left=100, color="lime", height=bar_height)

    ax.vlines(value, ymin=-bar_height/2, ymax=bar_height/2, color="black", linewidth=1.5)
    ax.text(value, bar_height/2 + 0.05, f"{value:.2f}", ha="center", va="bottom",
            fontsize=fs_val, color="black", fontweight="bold")

    ax.set_xlim(20, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_xticks([20, 90, 100, int(max_val)])
    ax.set_title(label, fontsize=fs_title, pad=fs_pad)

def color_bar_symmetry(ax, value, label, fs=None):
    bar_height = 0.1
    fs_val = fs if fs else 9
    fs_title = fs if fs else 12
    fs_pad = -8 if fs else -15
    min_val, max_val = 50, 150
    min_val = min(min_val, value - 10)
    max_val = max(max_val, value + 10)

    ax.barh(0, 90 - min_val, left=min_val, color="red", height=bar_height)
    ax.barh(0, 20, left=90, color="lime", height=bar_height)
    ax.barh(0, max_val - 110, left=110, color="red", height=bar_height)

    ax.vlines(value, ymin=-bar_height/2, ymax=bar_height/2, color="black", linewidth=1.5)
    ax.text(value, bar_height/2 + 0.05, f"{value:.2f}", ha="center", va="bottom",
            fontsize=fs_val, color="black", fontweight="bold")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_xticks([90, 100, 110])
    ax.set_title(label, fontsize=fs_title, pad=fs_pad)


def color_bar_double_support(ax, value, label, fs=None):
    bar_height = 0.1
    fs_val = fs if fs else 9
    fs_title = fs if fs else 12
    fs_pad = -8 if fs else -15
    max_val = 75

    ax.barh(0, 35, left=0, color="lime", height=bar_height)
    ax.barh(0, 3.5, left=35, color="gold", height=bar_height)
    ax.barh(0, max_val - 38.5, left=38.5, color="red", height=bar_height)

    ax.vlines(value, ymin=-bar_height/2, ymax=bar_height/2, color="black", linewidth=1.5)
    ax.text(value, bar_height/2 + 0.05, f"{value:.2f}%", ha="center", va="bottom",
            fontsize=fs_val, color="black", fontweight="bold")

    ax.set_xlim(0, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 35, 38.5, max_val])
    ax.set_title(label, fontsize=fs_title, pad=fs_pad)


def color_bar_speed(ax, value, mean, sd, label, k=3.0, pad_frac=0.06,
                    clinical_threshold=1.12, yellow_margin=0.22, fs=None):
    """
    Gait speed colorbar (Studenski 2011):
      ROJO:     < (clinical_threshold - yellow_margin)  → riesgo
      AMARILLO: (clinical_threshold - yellow_margin) → clinical_threshold → límite
      VERDE:    >= clinical_threshold  → bueno (sin límite superior)
    """
    bar_height = 0.1
    fs_val = fs if fs else 9
    fs_title = fs if fs else 12
    fs_pad = -8 if fs else -150

    if not np.isfinite(sd) or sd <= 0:
        sd = max(1e-6, abs(mean) * 0.05 + 0.10)

    thresh    = clinical_threshold        # 1.12
    yellow_lo = thresh - yellow_margin    # 1.02

    # Límites del eje
    x_min = 0.0
    x_max = max(mean + k * sd, float(value) * 1.08, thresh + 0.5)
    pad   = pad_frac * (x_max - x_min)
    x_min -= pad
    x_max += pad

    def seg(left, right, color):
        if right > left:
            ax.barh(0, right - left, left=left, color=color, height=bar_height)

    seg(x_min,     yellow_lo,  "red")   # rojo: riesgo
    seg(yellow_lo, thresh,     "gold")  # amarillo: zona límite
    seg(thresh,    x_max,      "lime")  # verde: bueno (todo lo que pase de 1.12)

    # Marcador del valor
    ax.vlines(float(value), ymin=-bar_height/2, ymax=bar_height/2,
              color="black", linewidth=1.8)
    ax.text(float(value), bar_height/2 + 0.05, f"{float(value):.2f}",
            ha="center", va="bottom", fontsize=9, color="black", fontweight="bold")

    ticks = sorted({round(x_min, 1), round(yellow_lo, 2),
                    round(thresh, 2), round(x_max, 1)})
    ax.set_xticks(ticks)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.30, 0.30)
    ax.set_yticks([])
    ax.set_title(label, fontsize=fs_title, pad=fs_pad)

def color_bar_pdkv(ax, value_right, value_left, label, fs=None):
    bar_height = 0.1
    fs_val = fs if fs else 9
    fs_title = fs if fs else 12
    fs_pad = -8 if fs else -15

    # Definir límites
    min_val = -15
    max_val = 15

    # Zona buena (-5 a 5) → verde
    ax.barh(0, 10, left=-5, color="lime", height=bar_height)  

    # Zona media (-10 a -5 y 5 a 10) → amarillo
    ax.barh(0, 5, left=-10, color="gold", height=bar_height)
    ax.barh(0, 5, left=5, color="gold", height=bar_height)

    # Zonas malas (<-10 y >10) → rojo
    ax.barh(0, 5, left=min_val, color="red", height=bar_height)
    ax.barh(0, 5, left=10, color="red", height=bar_height)

    # Línea y texto para el valor derecho (azul)
    ax.vlines(value_right, ymin=-bar_height/2, ymax=bar_height/2,
              color="blue", linewidth=1.5, label="R")
    ax.text(value_right, bar_height/2 + 0.06, f"{value_right:.2f}°",
            ha="center", va="bottom", fontsize=fs_val, color="blue", fontweight="bold")

    # Línea y texto para el valor izquierdo (gris)
    ax.vlines(value_left, ymin=-bar_height/2, ymax=bar_height/2,
              color="dimgrey", linewidth=1.5, label="L")
    ax.text(value_left, bar_height/2, f"{value_left:.2f}°",
            ha="center", va="bottom", fontsize=fs_val, color="dimgrey", fontweight="bold")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_xticks([min_val, -10, -5, 0, 5, 10, max_val])
    ax.set_title(label, fontsize=fs_title, pad=fs_pad)

def color_bar_footclearance(ax, value_right_mm, value_left_mm, label, age_years, fs=None):
    import numpy as np
    bar_height = 0.1
    fs_val = fs if fs else 9
    fs_title = fs if fs else 12
    fs_pad = -8 if fs else -15

    # --- Umbrales por edad (semáforo) ---
    def _fc_thresholds(age):
        if age is None or age < 18:
            group = "young"
        elif 18 <= age <= 35:
            group = "young"
        elif 36 <= age <= 59:
            group = "mid"
        elif 60 <= age <= 80:
            group = "older"
        else:
            group = "older"

        if group == "young":
            # Rojo: <9 | >24 ; Amarillo: 9–11 y 21–24 ; Verde: 12–20
            return dict(red_lo=0.0, ylo_lo=9.0, ylo_hi=11.0, g_lo=12.0, g_hi=20.0, yhi_lo=21.0, yhi_hi=24.0, red_hi=24.0)
        elif group == "mid":
            # Rojo: <8 | >22 ; Amarillo: 8–9 y 19–22 ; Verde: 10–18
            return dict(red_lo=0.0, ylo_lo=8.0, ylo_hi=9.0,  g_lo=10.0, g_hi=18.0, yhi_lo=19.0, yhi_hi=22.0, red_hi=22.0)
        else:  # older
            # Rojo: <6 | >18 ; Amarillo: 6–7 y 16–18 ; Verde: 8–15
            return dict(red_lo=0.0, ylo_lo=6.0, ylo_hi=7.0,  g_lo=8.0,  g_hi=15.0, yhi_lo=16.0, yhi_hi=18.0, red_hi=18.0)

    thr = _fc_thresholds(age_years)

    # Rango x dinámico (igual a tu lógica original)
    vmax = np.nanmax([value_right_mm, value_left_mm])
    if not np.isfinite(vmax):
        vmax = 25.0
    max_val = max(25.0, vmax * 1.3)

    # === Dibujar franjas contiguas, estilo de tus otras colorbars ===
    # Segmentos (sin huecos):
    # [0, ylo_lo] rojo  | [ylo_lo, g_lo] amarillo | [g_lo, g_hi] verde
    # [g_hi, yhi_hi] amarillo | [yhi_hi, max_val] rojo
    def _bar(left, right, color):
        left = max(0.0, float(left))
        right = min(float(max_val), float(right))
        width = right - left
        if width > 0:
            ax.barh(0, width, left=left, color=color, height=bar_height)

    _bar(thr["red_lo"], thr["ylo_lo"], "red")     # rojo bajo
    _bar(thr["ylo_lo"], thr["g_lo"],  "gold")     # amarillo bajo (pegado al verde)
    _bar(thr["g_lo"],  thr["g_hi"],  "lime")      # verde
    _bar(thr["g_hi"],  thr["yhi_hi"], "gold")     # amarillo alto (pegado al verde)
    _bar(thr["yhi_hi"], max_val,      "red")      # rojo alto hasta xlim

    # Línea y texto para Right (azul)
    ax.vlines(value_right_mm, ymin=-bar_height/2, ymax=bar_height/2,
              colors="blue", linewidth=2, label="R")
    ax.text(value_right_mm, bar_height/2, f"{value_right_mm:.1f}",
            ha="center", va="bottom", fontsize=fs_val, color="blue", fontweight="bold")

    # Línea y texto para Left (gris oscuro)
    ax.vlines(value_left_mm, ymin=-bar_height/2, ymax=bar_height/2,
              colors="dimgrey", linewidth=2, label="L")
    ax.text(value_left_mm, bar_height/2 + 0.06, f"{value_left_mm:.1f}",
            ha="center", va="bottom", fontsize=fs_val, color="dimgrey", fontweight="bold")

    # Línea de referencia suelo (0 mm)
    ax.vlines(0, ymin=-bar_height/2, ymax=bar_height/2,
              linestyle='--', linewidth=1, color="black")

    # Ejes (mismo estilo que ya usas)
    ax.set_xlim(0, max_val)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 5, 10, 20, int(max_val)])
    ax.set_title(label, fontsize=fs_title, pad=fs_pad)
