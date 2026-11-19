"""
MEAS_VS_SIM_POLAR.py  –  BYTECHLAB RF
======================================
Overlays measured S21 cuts (from the antenna positioner CSV files) with
CST Studio farfield simulation cuts on polar plots, one plot per frequency.

Files are discovered automatically:
  MEAS/  – scanned for CSVs containing "AZIM" or "ELEV" in the filename
  SIM/   – scanned for farfield TXTs; frequencies parsed from filenames

Two cut planes are supported:
  • Elevation cut  (Phi=90° / Phi=270°):  CONST_AZIM file, azim=0 fixed
  • Azimuth  cut  (Phi=0°  / Phi=180°):  CONST_ELEV file, elev=0 fixed

Coordinate conventions
-----------------------
  CST  Theta=0   → boresight  (our elev/azim = 0°)
  CST  Phi=90°   → elevation cut   (matches CONST_AZIM measurement)
  CST  Phi=0°    → azimuth cut     (matches CONST_ELEV measurement)

Normalization
--------------
  Measurement S21 is shifted (dB offset) so its global peak aligns with the
  simulation peak directivity, giving a shape-normalised overlay.

Rotation offset
---------------
  MEAS_ANGLE_OFFSET_DEG independently rotates the measurement trace on the
  polar plot without touching the simulation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob
from scipy.signal import savgol_filter

# ============================================================================
# ── USER CONFIGURATION  ──────────────────────────────────────────────────────
# ============================================================================

# Folders (relative or absolute)
MEAS_DIR = "MEAS"   # scanned automatically for AZIM / ELEV CSVs
SIM_DIR  = "SIM"    # scanned automatically for farfield TXTs

# Output directories
OUT_DIR_ELEV = "EXPORTED/PLOTS_ELEV_CUT"
OUT_DIR_AZIM = "EXPORTED/PLOTS_AZIM_CUT"

# Polar plot radial axis limits [dB]
GAIN_MIN_DB = -30
GAIN_MAX_DB =  20

# ── Rotation offset ──────────────────────────────────────────────────────────
# Angular offset (degrees) added to the MEASUREMENT angles before plotting.
# Compensates for a physical rotation between the simulated model orientation
# and the actual antenna mounting on the positioner.
# Positive = clockwise rotation of the measurement trace on the polar plot.
# Set to 0 to disable.
MEAS_ANGLE_OFFSET_DEG = 0   # e.g.  90  or  -45

# ── Measurement smoothing ─────────────────────────────────────────────────────
# Savitzky-Golay smoothing applied to the measurement trace before plotting.
# MEAS_SMOOTH_WINDOW : number of points in the filter window (must be odd).
#   Larger = smoother.  Set to 0 or 1 to disable.
#   With 5° angular step: window=5 ≈ 25° smoothing, window=9 ≈ 45°.
# MEAS_SMOOTH_POLYORDER : polynomial order (2 or 3 recommended, must be < window).
MEAS_SMOOTH_WINDOW    = 5   # set to 0 or 1 to disable
MEAS_SMOOTH_POLYORDER = 2

# ── Phi-plane assignment ──────────────────────────────────────────────────────
# Controls which CST Phi planes are mapped to the elevation / azimuth cuts.
#
#   SWAP_PHI_PLANES = False  (default)
#     Elevation cut → Phi = 90° / 270°
#     Azimuth  cut  → Phi =  0° / 180°
#
#   SWAP_PHI_PLANES = True
#     Elevation cut → Phi =  0° / 180°
#     Azimuth  cut  → Phi = 90° / 270°
#
# Set to True when the simulation model is rotated 90° around boresight
# relative to the physical antenna orientation on the positioner.
SWAP_PHI_PLANES = False

# ============================================================================
# ── COLOURED LOGGER  ─────────────────────────────────────────────────────────
# ============================================================================

class C:
    """ANSI colour codes."""
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREY   = "\033[90m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"

def log_header(msg):
    print(f"\n{C.BOLD}{C.CYAN}{'─'*60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {msg}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─'*60}{C.RESET}")

def log_section(msg):
    print(f"\n{C.BOLD}{C.WHITE}── {msg}{C.RESET}")

def log_info(msg):
    print(f"  {C.CYAN}ℹ{C.RESET}  {msg}")

def log_ok(msg):
    print(f"  {C.GREEN}✔{C.RESET}  {msg}")

def log_warn(msg):
    print(f"  {C.YELLOW}⚠{C.RESET}  {C.YELLOW}{msg}{C.RESET}")

def log_error(msg):
    print(f"\n  {C.RED}✘  ERROR: {msg}{C.RESET}\n")

def log_save(path):
    print(f"  {C.GREEN}💾{C.RESET}  {C.GREY}Saved:{C.RESET} {path}")

def log_freq(f_ghz):
    print(f"\n{C.BOLD}{C.MAGENTA}  ▶  {f_ghz} GHz{C.RESET}")

# ============================================================================
# ── AUTO-DISCOVERY  ──────────────────────────────────────────────────────────
# ============================================================================

def discover_meas_files(meas_dir):
    """
    Scan meas_dir for CSV files using the CONST_AZIM / CONST_ELEV naming scheme:
      CONST_AZIM = azimuth held constant, elevation varies → elevation cut
      CONST_ELEV = elevation held constant, azimuth varies → azimuth cut

    Returns (path_elev_cut, path_azim_cut).
    Raises RuntimeError if either is missing or ambiguous.
    """
    all_csvs = glob.glob(os.path.join(meas_dir, "*.csv"))
    elev_files = [f for f in all_csvs if "CONST_AZIM" in os.path.basename(f).upper()]
    azim_files = [f for f in all_csvs if "CONST_ELEV" in os.path.basename(f).upper()]

    def _pick(files, label):
        if len(files) == 0:
            raise RuntimeError(
                f"No measurement CSV with '{label}' in its name found in '{meas_dir}'."
            )
        if len(files) > 1:
            raise RuntimeError(
                f"Multiple '{label}' CSVs found in '{meas_dir}' – please keep only one:\n"
                + "\n".join(f"    {f}" for f in files)
            )
        return files[0]

    return _pick(elev_files, "CONST_AZIM"), _pick(azim_files, "CONST_ELEV")


def discover_sim_files(sim_dir):
    """
    Scan sim_dir for CST farfield TXT files matching the naming scheme:
      "farfield (f=<N>) [1].txt"   e.g. "farfield (f=3) [1].txt"

    Returns a sorted list of (freq_ghz: int, filepath: str) tuples.
    Raises RuntimeError if no matching files are found.
    """
    all_txts = glob.glob(os.path.join(sim_dir, "*.txt"))
    results = []
    # Match exactly:  farfield (f=<N>) [<anything>].txt
    # Accepts integer or decimal frequencies, e.g. f=6, f=5.8, f=5,8
    pattern = re.compile(r'farfield\s*\(f=(\d+(?:[.,]\d+)?)\)', re.IGNORECASE)
    for fpath in all_txts:
        name = os.path.basename(fpath)
        m = pattern.search(name)
        if m:
            f = float(m.group(1).replace(',', '.'))
            # keep whole numbers as int so labels/filenames read "6 GHz", not "6.0 GHz"
            results.append((int(f) if f.is_integer() else f, fpath))

    if not results:
        raise RuntimeError(
            f"No CST farfield TXT files found in '{sim_dir}'.\n"
            f"  Expected filenames like: 'farfield (f=3) [1].txt'"
        )

    results.sort(key=lambda x: x[0])
    return results


# ============================================================================
# ── FREQUENCY VALIDATION  ────────────────────────────────────────────────────
# ============================================================================

def validate_frequencies(sim_freqs_ghz, meas_freqs_hz):
    """
    Check that every simulation frequency (integer GHz) falls within the
    measurement frequency range.  Raises RuntimeError with a clear explanation
    if any simulation frequency is outside the measurement range.
    """
    meas_min_ghz = meas_freqs_hz[0]  / 1e9
    meas_max_ghz = meas_freqs_hz[-1] / 1e9

    out_of_range = [f for f in sim_freqs_ghz
                    if f < meas_min_ghz - 0.005 or f > meas_max_ghz + 0.005]

    if out_of_range:
        raise RuntimeError(
            f"Simulation frequenc{'y' if len(out_of_range)==1 else 'ies'} "
            f"{out_of_range} GHz "
            f"{'is' if len(out_of_range)==1 else 'are'} outside the measurement "
            f"range ({meas_min_ghz:.3f} – {meas_max_ghz:.3f} GHz).\n\n"
            f"  Simulation files cover: {sim_freqs_ghz} GHz\n"
            f"  Measurement covers:     {meas_min_ghz:.3f} – {meas_max_ghz:.3f} GHz\n\n"
            f"  → Check that the SIM and MEAS folders belong to the same campaign."
        )


# ============================================================================
# ── LOAD MEASUREMENT DATA  ───────────────────────────────────────────────────
# ============================================================================

def load_meas_csv(path):
    """
    Parse a BYTECHLAB positioner 2-D cut CSV.
    Returns (freqs_hz, angles_deg, s21_matrix)
      freqs_hz   : 1-D array, length N_f
      angles_deg : 1-D array, length N_a  (sorted ascending)
      s21_matrix : 2-D array, shape (N_a, N_f)
    """
    df = pd.read_csv(path)
    freqs_hz = np.fromstring(df["frequencies"].iloc[0], sep=" ")

    # Auto-detect the sweep axis (the column that varies)
    if df["azim_deg"].nunique() == 1:
        angle_col = "elev_deg"
    else:
        angle_col = "azim_deg"

    angles_deg = df[angle_col].values
    s21_matrix = np.vstack([np.fromstring(row, sep=" ") for row in df["s21_logmag"]])

    sort_idx = np.argsort(angles_deg)
    angles_deg = angles_deg[sort_idx]
    s21_matrix = s21_matrix[sort_idx]

    return freqs_hz, angles_deg, s21_matrix


def get_meas_at_freq(freqs_hz, angles_deg, s21_matrix, target_hz):
    """Return (angles, s21_column, actual_freq_GHz) for the nearest frequency."""
    idx = int(np.argmin(np.abs(freqs_hz - target_hz)))
    return angles_deg, s21_matrix[:, idx], freqs_hz[idx] / 1e9


def smooth_meas(angles_deg, s21):
    """
    Apply Savitzky-Golay smoothing to a measurement cut, then close the
    wrap-around seam by appending a copy of the -180° point at +180°.

    Closing happens AFTER smoothing so the duplicated endpoint is always
    identical to the already-smoothed first point — eliminating any gap.
    """
    w = MEAS_SMOOTH_WINDOW
    if w > 1:
        if w % 2 == 0:
            w += 1
        p = min(MEAS_SMOOTH_POLYORDER, w - 1)
        pad = w // 2 + 1
        padded   = np.concatenate([s21[-pad:], s21, s21[:pad]])
        smoothed = savgol_filter(padded, window_length=w, polyorder=p)
        s21 = smoothed[pad : pad + len(s21)]

    # Close the wrap-around gap AFTER smoothing
    if angles_deg[0] == -180.0 and angles_deg[-1] < 180.0:
        angles_deg = np.append(angles_deg, 180.0)
        s21        = np.append(s21, s21[0])

    return angles_deg, s21


# ============================================================================
# ── LOAD SIMULATION DATA  ────────────────────────────────────────────────────
# ============================================================================

def load_sim_file(fpath):
    """
    Load a CST farfield export (.txt, two header lines, whitespace-separated).
    Returns a DataFrame.
    """
    df = pd.read_csv(
        fpath, skiprows=2, sep=r"\s+",
        names=["Theta", "Phi", "Abs_Dir", "Abs_Theta",
               "Phase_Theta", "Abs_Phi", "Phase_Phi", "Ax_Ratio"],
        engine="python",
    )
    return df


def sim_full360_cut(df_sim, phi_main, phi_back):
    """
    Reconstruct a full ±180° polar cut from two complementary CST Phi planes.
    Returns (plot_angles_deg, directivity_dBi) sorted -180 → +180.
    """
    front = df_sim[np.isclose(df_sim["Phi"], phi_main, atol=0.5)]
    back  = df_sim[np.isclose(df_sim["Phi"], phi_back,  atol=0.5)]

    theta_f = front["Theta"].values
    dir_f   = front["Abs_Dir"].values

    back_mask = back["Theta"].values > 0
    theta_b = -back["Theta"].values[back_mask]
    dir_b   =  back["Abs_Dir"].values[back_mask]

    angles = np.concatenate([theta_b, theta_f])
    dirs   = np.concatenate([dir_b,   dir_f])
    sort_i = np.argsort(angles)
    return angles[sort_i], dirs[sort_i]


# ============================================================================
# ── PATTERN METRICS  ─────────────────────────────────────────────────────────
# ============================================================================

def compute_rms(sim_angles_deg, sim_dBi,
                meas_angles_deg, meas_dBi):
    """
    RMS difference between simulation and measurement in the dB domain.
    Both curves are interpolated onto a common 1° grid and clipped to
    GAIN_MIN_DB so the noise floor does not distort the result.
    Lower is better (0 dB = perfect match).
    """
    grid = np.arange(-180, 181, 1.0)
    db_sim  = np.interp(grid, sim_angles_deg,  np.clip(sim_dBi,  GAIN_MIN_DB, None))
    db_meas = np.interp(grid, meas_angles_deg, np.clip(meas_dBi, GAIN_MIN_DB, None))
    return float(np.sqrt(np.mean((db_sim - db_meas) ** 2)))


# ============================================================================
# ── POLAR PLOT  ──────────────────────────────────────────────────────────────
# ============================================================================

def polar_plot(sim_angles_deg, sim_dir_dBi,
               meas_angles_deg, meas_s21_norm_dBi,
               title, save_path, axis_label="Angle [°]",
               rms_db=None):
    """
    Overlay polar plot: simulation (dashed orange) + measurement (solid blue).
    0° at top, angles increase clockwise.
    If rms_db is provided, a colour-coded RMS Error badge is placed in the
    bottom-right corner (green ≤3 dB, amber 3–6 dB, red >6 dB).
    """
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})

    ax.plot(np.deg2rad(sim_angles_deg),  sim_dir_dBi,
            label="Simulation",  lw=2,   ls="--", color="darkorange")
    ax.plot(np.deg2rad(meas_angles_deg), meas_s21_norm_dBi,
            label="Measurement", lw=1.8,            color="steelblue")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_rlim(GAIN_MIN_DB, GAIN_MAX_DB)
    ax.set_rticks(np.arange(GAIN_MIN_DB, GAIN_MAX_DB + 1, 10))

    ax.grid(True, ls="--", alpha=0.6, color="gray")
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])

    ax.set_title(title, pad=22, fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.12), fontsize=11)

    # ── RMS Error badge – bottom-right corner ─────────────────────────
    if rms_db is not None:
        if rms_db <= 3.0:
            rms_color = "#2ecc71"   # green
        elif rms_db <= 6.0:
            rms_color = "#f39c12"   # amber
        else:
            rms_color = "#e74c3c"   # red
        fig.text(
            0.97, 0.10,
            "RMS Error\n" + f"{rms_db:.1f} dB",
            ha="right", va="center", fontsize=10, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=rms_color,
                      edgecolor="white", linewidth=1.2, alpha=0.92),
            transform=fig.transFigure,
        )

    fig.text(0.5, 0.03, axis_label, ha="center", va="center",
             fontsize=12, fontweight="normal")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log_save(save_path)
    plt.close(fig)


# ============================================================================
# ── MAIN  ────────────────────────────────────────────────────────────────────
# ============================================================================

def main():
    log_header("BYTECHLAB RF  –  Meas vs Sim Polar Comparison")

    os.makedirs(OUT_DIR_ELEV, exist_ok=True)
    os.makedirs(OUT_DIR_AZIM, exist_ok=True)

    # ── Phi plane assignments ──────────────────────────────────────────
    phi_elev_main, phi_elev_back = (90, 270) if not SWAP_PHI_PLANES else (0, 180)
    phi_azim_main, phi_azim_back = (0, 180)  if not SWAP_PHI_PLANES else (90, 270)
    phi_elev_label = f"Phi = {phi_elev_main}°"
    phi_azim_label = f"Phi = {phi_azim_main}°"

    # ── Discover measurement files ─────────────────────────────────────
    log_section("Discovering measurement files")
    try:
        csv_elev_cut, csv_azim_cut = discover_meas_files(MEAS_DIR)
    except RuntimeError as e:
        log_error(str(e))
        return

    log_ok(f"Elevation cut  (ELEV): {csv_elev_cut}")
    log_ok(f"Azimuth  cut  (AZIM):  {csv_azim_cut}")

    # ── Discover simulation files ──────────────────────────────────────
    log_section("Discovering simulation files")
    try:
        sim_entries = discover_sim_files(SIM_DIR)
    except RuntimeError as e:
        log_error(str(e))
        return

    sim_freqs_ghz = [f for f, _ in sim_entries]
    log_ok(f"Found {len(sim_entries)} simulation file(s)")
    for f_ghz, fpath in sim_entries:
        log_info(f"{f_ghz} GHz  →  {os.path.basename(fpath)}")
    log_info(f"Frequency list: {sim_freqs_ghz} GHz")

    # ── Load measurement data ──────────────────────────────────────────
    log_section("Loading measurement data")
    elev_freqs, elev_angles, elev_s21 = load_meas_csv(csv_elev_cut)
    azim_freqs, azim_angles, azim_s21 = load_meas_csv(csv_azim_cut)

    log_ok(f"Elevation cut:  {len(elev_angles)} angles,  "
           f"{len(elev_freqs)} freqs  "
           f"({elev_freqs[0]/1e9:.3f} – {elev_freqs[-1]/1e9:.3f} GHz)")
    log_ok(f"Azimuth  cut:   {len(azim_angles)} angles,  "
           f"{len(azim_freqs)} freqs  "
           f"({azim_freqs[0]/1e9:.3f} – {azim_freqs[-1]/1e9:.3f} GHz)")

    # ── Validate frequency overlap ─────────────────────────────────────
    log_section("Validating frequency coverage")
    try:
        validate_frequencies(sim_freqs_ghz, elev_freqs)
        validate_frequencies(sim_freqs_ghz, azim_freqs)
    except RuntimeError as e:
        log_error(str(e))
        return
    log_ok("All simulation frequencies are within the measurement range  ✔")

    # ── Settings summary ───────────────────────────────────────────────
    log_section("Settings")
    log_info(f"Phi planes       : {'SWAPPED' if SWAP_PHI_PLANES else 'default'}  "
             f"(Elev={phi_elev_label}, Azim={phi_azim_label})")
    log_info(f"Angle offset     : {MEAS_ANGLE_OFFSET_DEG:+.1f}°"
             + ("  (disabled)" if MEAS_ANGLE_OFFSET_DEG == 0 else ""))
    if MEAS_SMOOTH_WINDOW <= 1:
        log_info("Smoothing        : disabled")
    else:
        w = MEAS_SMOOTH_WINDOW + (1 if MEAS_SMOOTH_WINDOW % 2 == 0 else 0)
        log_info(f"Smoothing        : Savitzky-Golay  window={w}  polyorder={MEAS_SMOOTH_POLYORDER}  "
                 f"(≈ {w * 5}° span)")
    log_info(f"Radial axis      : {GAIN_MIN_DB} … {GAIN_MAX_DB} dB")
    log_info(f"Output (elev)    : {OUT_DIR_ELEV}")
    log_info(f"Output (azim)    : {OUT_DIR_AZIM}")

    # ── Per-frequency plots ────────────────────────────────────────────
    log_section("Generating plots")
    offset_note = (f", meas offset {MEAS_ANGLE_OFFSET_DEG:+.0f}°"
                   if MEAS_ANGLE_OFFSET_DEG != 0 else "")

    for f_ghz, sim_fpath in sim_entries:
        log_freq(f_ghz)

        log_info(f"Loading sim file: {os.path.basename(sim_fpath)}")
        df_sim = load_sim_file(sim_fpath)
        log_info(f"Sim data shape: {df_sim.shape[0]} rows  "
                 f"(Theta {df_sim['Theta'].min():.0f}°–{df_sim['Theta'].max():.0f}°,  "
                 f"{df_sim['Phi'].nunique()} Phi planes)")

        # ── Elevation-plane cut ────────────────────────────────────────
        sim_ang_e, sim_dir_e = sim_full360_cut(df_sim, phi_elev_main, phi_elev_back)

        meas_ang_e, meas_s21_e, actual_f_e = get_meas_at_freq(
            elev_freqs, elev_angles, elev_s21, f_ghz * 1e9)

        peak_sim_e  = np.nanmax(sim_dir_e)
        peak_meas_e = np.nanmax(meas_s21_e)
        norm_shift_e = peak_sim_e - peak_meas_e
        meas_ang_e, meas_norm_e = smooth_meas(
            meas_ang_e, meas_s21_e + norm_shift_e)
        meas_ang_e_rot = meas_ang_e + MEAS_ANGLE_OFFSET_DEG

        rms_e = compute_rms(sim_ang_e, sim_dir_e, meas_ang_e_rot, meas_norm_e)
        log_info(f"Elev cut  ({phi_elev_label}):  "
                 f"sim peak {peak_sim_e:.1f} dBi,  "
                 f"meas peak {peak_meas_e:.1f} dB,  "
                 f"norm shift {norm_shift_e:+.1f} dB,  "
                 f"RMS error {rms_e:.1f} dB")

        title_e = (f"Antenna Radiation Pattern  {f_ghz} GHz  –  Elevation Cut ({phi_elev_label})\n"
                   f"[Meas @ {actual_f_e:.3f} GHz, norm. to sim peak "
                   f"{peak_sim_e:.1f} dBi{offset_note}]")
        fname_e = f"{f_ghz}GHz_{phi_elev_label.replace(' ','').replace('=','').replace('°','deg')}_elev.png"
        polar_plot(
            sim_ang_e,      sim_dir_e,
            meas_ang_e_rot, meas_norm_e,
            title_e,
            os.path.join(OUT_DIR_ELEV, fname_e),
            axis_label="Elevation Angle [°]",
            rms_db=rms_e,
        )

        # ── Azimuth-plane cut ──────────────────────────────────────────
        sim_ang_a, sim_dir_a = sim_full360_cut(df_sim, phi_azim_main, phi_azim_back)

        meas_ang_a, meas_s21_a, actual_f_a = get_meas_at_freq(
            azim_freqs, azim_angles, azim_s21, f_ghz * 1e9)

        peak_sim_a  = np.nanmax(sim_dir_a)
        peak_meas_a = np.nanmax(meas_s21_a)
        norm_shift_a = peak_sim_a - peak_meas_a
        meas_ang_a, meas_norm_a = smooth_meas(
            meas_ang_a, meas_s21_a + norm_shift_a)
        meas_ang_a_rot = meas_ang_a + MEAS_ANGLE_OFFSET_DEG

        rms_a = compute_rms(sim_ang_a, sim_dir_a, meas_ang_a_rot, meas_norm_a)
        log_info(f"Azim cut  ({phi_azim_label}):   "
                 f"sim peak {peak_sim_a:.1f} dBi,  "
                 f"meas peak {peak_meas_a:.1f} dB,  "
                 f"norm shift {norm_shift_a:+.1f} dB,  "
                 f"RMS error {rms_a:.1f} dB")

        title_a = (f"Antenna Radiation Pattern  {f_ghz} GHz  –  Azimuth Cut ({phi_azim_label})\n"
                   f"[Meas @ {actual_f_a:.3f} GHz, norm. to sim peak "
                   f"{peak_sim_a:.1f} dBi{offset_note}]")
        fname_a = f"{f_ghz}GHz_{phi_azim_label.replace(' ','').replace('=','').replace('°','deg')}_azim.png"
        polar_plot(
            sim_ang_a,      sim_dir_a,
            meas_ang_a_rot, meas_norm_a,
            title_a,
            os.path.join(OUT_DIR_AZIM, fname_a),
            axis_label="Azimuth Angle [°]",
            rms_db=rms_a,
        )

    log_header(f"Done – {len(sim_entries) * 2} plots saved")


if __name__ == "__main__":
    main()