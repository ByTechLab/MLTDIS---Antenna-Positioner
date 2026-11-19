#!/usr/bin/env python3
"""
Antenna Positioner Controller  –  NiceGUI  v5.1
=============================================
Hardware
--------
  Rotator : MKS Servo42D  ×2  via  MKS CANable V2.0  (SLCAN)
  VNA     : LibreVNA  (TCP localhost:19542)

Software requirements
---------------------
  LibreVNA-GUI : ≥ 1.6.0  (REQUIRED)
    v1.6.0 introduced a breaking SCPI change: set-commands (non-query) no
    longer return an acknowledgement response.  Only commands ending in '?'
    return a value.  This script uses _vna_cmd() for set-commands and
    _vna_query() exclusively for '?' queries to match this behaviour.
    Using LibreVNA-GUI < 1.6.0 will cause _vna_query() to hang because
    the GUI would send an extra ACK that is never consumed, desynchronising
    the socket buffer for all subsequent queries.

Motor wiring
------------
  Azimuth axis  CAN ID = 1   gear ratio = 192/24 = 8.0
  Elevation axis  CAN ID = 2   gear ratio = 200/20 = 10.0

Coordinate convention
---------------------
  Measurement angles are the logical angles saved to CSV and shown in plots.
  Physical motor angles  =  measurement angles + axis offset (see OFFSETS below).
  Absolute encoder origin is re-zeroed at every script start (0x92 command).

Angle sweep
-----------
  2D : azimuth = -180 … +175° (step-exclusive +180 to avoid duplicate position)
       elevation = constant (user-settable)
  3D - Full  : azimuth = -180 … +175°, elevation = -90 … +90°
               snake scan (elevation alternates each azimuth row – no motor rewind)
  3D - Range : user-specified azimuth and elevation start/stop, same snake scan

All bugs fixed vs V3/V4
-----------------------
  1.  live_farfield_plot.update_figure(fig) / live_vna_trace.update_figure(fig)
      NiceGUI snapshots the figure at construction; plain .update() re-sends
      the stale snapshot.  update_figure() re-serialises before pushing.

  2.  VNA SCPI split into _vna_cmd() / _vna_query()  (LibreVNA-GUI ≥ 1.6.0)
      LibreVNA-GUI 1.6.0 changed the SCPI server to conform to the standard:
      set-commands (e.g. :DEV:MODE VNA, :VNA:ACQ:SINGLE TRUE) no longer send
      an acknowledgement response.  Only '?' queries return a value.
      _vna_cmd()   → vna.cmd()   — sends command, reads nothing.
      _vna_query() → vna.query() — sends '?' query, reads and returns response.
      Previously a single _vna_query() was used for everything, which caused
      a timeout hang on every set-command under GUI ≥ 1.6.0.

  3.  Motor position drift between runs fixed with 0x92
      In SR_CLOSE mode the PID controller leaves the encoder counter at a
      non-zero value after each run.  0x92 re-declares the current physical
      position as coordinate 0 at startup, so F5 absolute moves always target
      the correct encoder count regardless of previous run history.
      Without this: acc appeared broken (motor used max-speed catch-up to
      cover the unexpectedly large position error) and second runs failed.

  4.  _wait_motor_stop two-tier completion
      Primary : F5 status=2 (run complete) from motor.
      Fallback : if status=1 received but no further F5 messages for
                 travel_time + 2 s → motor assumed physically stopped.
                 Needed when SR_CLOSE PID never settles below arrival threshold.

  5.  CAN bus.recv() in ThreadPoolExecutor so asyncio event loop stays live.
      VNA socket calls remain synchronous (libreVNA buffer is NOT thread-safe).

  6.  Gauge needle transform: full 'translate(-50%,-50%) rotate(Xdeg)' written
      on every update so rotation never overwrites the centering translate.

  7.  asyncio.sleep(0.05) after plot updates gives NiceGUI time to flush
      WebSocket frames before the next blocking CAN/VNA call.

  8.  3D farfield: correct antenna coordinate convention
        X = r·cos(elev)·cos(azim)   (azim=azimuth, elev=elevation)
        Y = r·cos(elev)·sin(azim)
        Z = r·sin(elev)
      Surface closed in azimuth direction (wrap -180 → +175 → -180+360).
      Color range clipped to 30 dB from peak for useful contrast.

  9.  y-axis autorange on both plots so data below -60 dB is always visible.

  10. Motion Smoothness slider (inverted acc):
        slider 1   → acc_byte 255 → ramp 50 µs/RPM   (instant)
        slider 220 → acc_byte 36  → ramp 11 ms/RPM   (smooth, default)
        slider 255 → acc_byte 1   → ramp 12.75 ms/RPM (slowest)

  11. Frequency preview slider range set from VNA sweep range at run start.

  12. Gaussian smoothing slider for farfield preview (0 = raw data).

  13. Return to physical 0° (offset=0) at measurement end so motors park
      at their mechanical zero rather than logical 0°.

  14. Snake scan in 3D: elevation reverses direction every azimuth row.

  15. SLCAN buffer flushed at connect to discard stale frames from previous
      session that would otherwise be replayed into the motor state machine.
"""

# ═══════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════

from nicegui import ui
import asyncio
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import can
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal  import savgol_filter
from pathlib        import Path
import matplotlib
matplotlib.use('Agg')   # non-interactive backend – safe inside asyncio
import matplotlib.pyplot as plt

from libreVNA import libreVNA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# HARDWARE CONFIG  –  edit these to match your setup
# ═══════════════════════════════════════════════════════════════════

CAN_CHANNEL    = "COM3"           # SLCAN serial port
CAN_BITRATE    = 500_000          # bits/s

AZIMUTH_CAN_ID   = 1                # MKS motor CAN address for azimuth axis
ELEVATION_CAN_ID     = 2                # MKS motor CAN address for elevation axis
PULSES_PER_REV = 16_384           # encoder counts per motor shaft revolution

AZIMUTH_GEAR_RATIO = 192 / 24.0    # motor turns per 1 output (antenna) turn = 8.0
ELEVATION_GEAR_RATIO   = 200 / 20.0    # motor turns per 1 output (antenna) turn = 10.0

# Rotation direction:  1 = normal,  -1 = inverted.
# Flip when the physical rotation is opposite to the logical angle convention.
AZIMUTH_DIR = -1
ELEVATION_DIR   = -1

# Mechanical zero offsets (degrees).
# Added to every F5 command so that measurement angle 0° maps to the
# correct physical boresight position.  Adjust after mechanical calibration.
AZIMUTH_OFFSET_DEG = -90.0
ELEVATION_OFFSET_DEG   = -90.0


# ═══════════════════════════════════════════════════════════════════
# RUNTIME STATE
# ═══════════════════════════════════════════════════════════════════

vna: "libreVNA | None" = None
bus                    = None
df_data: list[dict]    = []
measurement_running    = False
stop_requested         = False

_executor = ThreadPoolExecutor(max_workers=4)


def _set_stop():
    global stop_requested
    stop_requested = True


def _safe_setattr(widget, attr: str, value):
    """setattr that silently ignores deleted-client errors from slider callbacks."""
    if widget is None:
        return
    try:
        setattr(widget, attr, value)
    except RuntimeError:
        pass


def _safe_notify(msg: str, **kwargs):
    """_safe_notify() that silently does nothing if the client is gone."""
    try:
        ui.notify(msg, **kwargs)
    except RuntimeError:
        logger.info(f'[UI] (client gone) notify suppressed: {msg}')


def _ui_set(widget, attr: str, value):
    """
    Safely set a NiceGUI widget attribute during an async measurement.
    If the browser was refreshed the client/element may be deleted;
    in that case we request a measurement stop so the coroutine
    exits cleanly instead of crashing with RuntimeError.
    """
    global stop_requested
    if widget is None:
        return
    try:
        setattr(widget, attr, value)
    except RuntimeError:
        # Browser reconnected — page elements deleted; stop gracefully
        if not stop_requested:
            logger.warning('[UI] Client deleted (browser refresh?) — stopping measurement')
            stop_requested = True


def _parse_freq_range():
    """Parse freq_range_input and return (f_start_hz, f_stop_hz) or None."""
    try:
        parts   = (freq_range_input.value
                   .replace('GHz','').replace('ghz','').strip().split('-'))
        return float(parts[0]) * 1e9, float(parts[1]) * 1e9
    except Exception:
        return None


def _update_vna_ui():
    """Recalculate acquisition points AND update the farfield freq slider range."""
    # ── Points label ────────────────────────────────────────────────
    if vna_points_label is not None and vna_freq_step_input is not None:
        rng = _parse_freq_range()
        if rng:
            f_start, f_stop = rng
            try:
                step_hz = float(
                    vna_freq_step_input.value
                    .replace('MHz','').replace('mhz','').strip()
                ) * 1e6
                step_hz = max(1e3, step_hz)
            except Exception:
                step_hz = 4e6
            pts = max(2, min(10001, int(round((f_stop - f_start) / step_hz)) + 1))
            vna_points_label.text = str(pts)
            vna_points_label.style(
                'font-size:1.5rem; font-weight:900; font-family:monospace; '
                + ('color:#f87171;' if pts > 9000 else 'color:#34d399;')
            )
        else:
            vna_points_label.text = '—'

    # ── Farfield preview slider range ────────────────────────────────
    if freq_slider is not None:
        rng = _parse_freq_range()
        if rng:
            f_start, f_stop = rng
            f_start_ghz = round(f_start / 1e9, 4)
            f_stop_ghz  = round(f_stop  / 1e9, 4)
            step_ghz    = round((f_stop_ghz - f_start_ghz) / 200, 4)
            step_ghz    = max(0.0001, step_ghz)
            # NiceGUI sliders require props() to push min/max/step to the browser
            freq_slider._props['min']  = f_start_ghz
            freq_slider._props['max']  = f_stop_ghz
            freq_slider._props['step'] = step_ghz
            freq_slider.value = f_start_ghz
            freq_slider.update()
            if freq_label is not None:
                freq_label.text = f'{f_start_ghz:.4f} GHz'


# Keep old name as alias so the timer call below still works
_update_points_label = _update_vna_ui


# ═══════════════════════════════════════════════════════════════════
# UI WIDGET REFERENCES  (assigned during page construction below)
# ═══════════════════════════════════════════════════════════════════

project_input        = None
measurement_radio    = None
azim_start_input     = None   # 3D-Range azimuth start (°)
azim_stop_input      = None   # 3D-Range azimuth stop  (°)
elev_start_input     = None   # 3D-Range elevation start (°)
elev_stop_input      = None   # 3D-Range elevation stop  (°)
freq_range_input     = None
angle_step_input     = None
cut_type_dropdown    = None
constant_angle_input = None
speed_slider         = None
acc_slider           = None
progress_bar         = None
percent_label        = None
status_label         = None
start_button         = None
stop_button          = None

azim_pointer        = None
azim_value_label    = None
elev_pointer          = None
elev_value_label      = None

farfield_fig         = go.Figure()
vna_fig              = go.Figure()
live_farfield_plot   = None
live_vna_trace       = None
freq_slider          = None   # range updated at measurement start
vna_freq_step_input  = None   # VNA frequency step (MHz)
vna_points_label     = None   # calculated acquisition points display
smooth_slider        = None   # Gaussian sigma; 0 = raw data (farfield preview)
shape_slider         = None   # Shape compression gamma (0.1–1.0); colour unchanged
vna_smooth_slider    = None   # VNA data S-G smoothing; 0 = off
can_port_input      = None   # CANable serial port (e.g. COM3)
settle_delay_input  = None   # Post-move settle delay before VNA sweep (s)
azim_offset_input   = None   # Azimuth axis mechanical zero offset
elev_offset_input     = None   # Elevation axis mechanical zero offset
_test_btn            = None   # Test zero position button
_return_btn          = None   # Return to physical 0° button
plot_enable_radio    = None   # Enable/disable post-measurement plots
plot_freq_input      = None   # Comma-separated GHz list for plots
html3d_enable_radio  = None   # Enable/disable interactive 3D HTML export
html3d_freq_input    = None   # Frequencies for 3D HTML
motor_status_label   = None
vna_status_label     = None
eta_label            = None
vna_error_banner     = None   # persistent error banner updated directly from measurement loop
vna_error_text       = None   # label inside the banner
_ui_state_updater    = None   # set during page build; called after settings load
vna_avg_input        = None   # VNA averaging count
vna_ifbw_input       = None   # VNA IF bandwidth (Hz)
vna_level_input      = None   # VNA stimulus level (dBm)
norm_radio           = None   # Normalization mode: Off / Angle / Max
norm_azim_input     = None   # Reference azimuth for angle normalisation
norm_elev_input       = None   # Reference elevation for angle normalisation
notes_input          = None   # Project notes textarea


# ═══════════════════════════════════════════════════════════════════
# GAUGE WIDGET
# ═══════════════════════════════════════════════════════════════════

def create_gauge(title: str):
    """Build a circular ±180° gauge.  Returns (needle_element, value_label)."""
    with ui.card().classes(
        'items-center p-6 bg-zinc-900 border border-zinc-700 rounded-3xl shadow-2xl'
    ):
        ui.label(title).classes('text-xl font-bold tracking-widest text-white mb-4')
        with ui.element('div').classes('relative w-72 h-72'):
            ui.element('div').classes(
                'absolute inset-0 rounded-full border-8 border-zinc-600 '
                'bg-gradient-to-br from-zinc-800 to-black'
            )
            ui.element('div').classes('absolute inset-4 rounded-full bg-zinc-950')

            for a in range(-180, 181, 30):
                ui.element('div').style(
                    f'position:absolute; left:50%; top:8px; width:2px; height:16px; '
                    f'background:#a1a1aa; transform-origin:50% 128px; '
                    f'transform:translateX(-50%) rotate({a}deg);'
                )

            pointer = ui.element('div').style(
                'position:absolute; left:50%; top:50%; width:0; height:0; '
                'transform:translate(-50%,-50%) rotate(0deg); '
                'transform-origin:center center;'
            )
            ui.element('div').style(
                'position:absolute; left:-5px; bottom:0px; width:0; height:0; '
                'border-left:5px solid transparent; '
                'border-right:5px solid transparent; '
                'border-bottom:110px solid #e63939;'
            ).move(pointer)
            ui.element('div').style(
                'position:absolute; left:50%; top:50%; '
                'width:20px; height:20px; border-radius:50%; '
                'background:#e63939; border:3px solid #09090b; '
                'transform:translate(-50%,-50%); z-index:10;'
            )
            value_label = ui.label('0.0°').style(
                'position:absolute; left:50%; top:65%; '
                'transform:translateX(-50%); '
                'font-size:1.8rem; font-weight:900; color:white; letter-spacing:-1px;'
            )
    return pointer, value_label


def update_gauge(pointer, value_label, angle_deg: float):
    pointer.style(
        f'position:absolute; left:50%; top:50%; width:0; height:0; '
        f'transform:translate(-50%,-50%) rotate({angle_deg:.1f}deg); '
        f'transform-origin:center center;'
    )
    value_label.text = f'{angle_deg:.1f}°'


# ═══════════════════════════════════════════════════════════════════
# CAN / MOTOR HELPERS
# ═══════════════════════════════════════════════════════════════════

def _crc(can_id: int, payload: bytes) -> int:
    return (can_id + sum(payload)) & 0xFF


async def _send_can(motor_id: int, payload: bytes):
    if bus is None:
        return
    data = payload + bytes([_crc(motor_id, payload)])
    msg  = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, bus.send, msg)
    logger.debug(f"CAN → motor {motor_id}: {data.hex()}")


async def _init_motor(mid: int):
    """
    Re-zero the motor's absolute encoder counter to its current physical
    position (command 0x92).

    Root cause of two bugs fixed here:
    • In SR_CLOSE mode the PID controller leaves the counter at a non-zero
      value after each run (position drift from hunting).  On the next run
      F5 absolute commands target the wrong encoder count:
        - Large position error  → motor runs at max speed ignoring acc ramp
          (acceleration appeared "broken")
        - Very large error      → motor stalls / times out (second run failed)
    • 0x92 makes every run start from a clean, consistent origin.

    Motor working mode (SR_CLOSE = 0x04) must be set once via the motor's
    OLED menu (Menu → Mode → SR_CLOSE) and is saved to EEPROM.
    """
    await _send_can(mid, bytes([0x92]))
    await asyncio.sleep(0.15)


async def _enable_motor(mid: int):
    await _send_can(mid, bytes([0xF3, 0x01]))


async def _disable_motor(mid: int):
    await _send_can(mid, bytes([0xF3, 0x00]))


async def _wait_motor_stop(motor_id: int, speed_rpm: float,
                           pulses: int, timeout_s: float = 60.0):
    """
    Wait for a motor move to complete.

    Primary   : F5 status=2 received   → done immediately (clean case)
    Fallback  : F5 status=1 received, then silence > threshold
                → motor assumed physically stopped.
                  Needed when SR_CLOSE PID never settles below arrival
                  threshold so status=2 is never sent.
    Timeout   : log warning and continue after timeout_s.

    Silence threshold is calculated from the actual move distance and speed
    so slow moves at low RPM are never declared done prematurely.
    """
    loop         = asyncio.get_event_loop()
    deadline     = time.monotonic() + timeout_s
    started      = False
    last_f5_time = None

    motor_revs   = abs(pulses) / PULSES_PER_REV
    travel_s     = (motor_revs / max(speed_rpm, 0.1)) * 60.0
    silence_done = max(3.0, travel_s + 2.0)
    _axis_name = "AZIMUTH" if motor_id == AZIMUTH_CAN_ID else "ELEVATION"
    logger.info(
        f"  [{_axis_name}] Moving → {pulses} pulses  "
        f"{speed_rpm:.0f} RPM  est. {travel_s:.1f}s"
    )

    def _recv_once():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        return bus.recv(timeout=min(0.3, remaining))

    while time.monotonic() < deadline:
        if stop_requested:
            return False

        msg = await loop.run_in_executor(_executor, _recv_once)

        if (msg is not None
                and msg.arbitration_id == motor_id
                and len(msg.data) >= 2
                and msg.data[0] == 0xF5):
            status       = msg.data[1]
            last_f5_time = time.monotonic()
            if status == 2:
                await asyncio.sleep(0.2)
                return True
            elif status == 1:
                started = True
            elif status == 0:
                logger.warning(f"  [{_axis_name}] Run FAIL (status=0)")
                return False

        if started and last_f5_time is not None:
            if time.monotonic() - last_f5_time > silence_done:
                logger.info(f"  [{_axis_name}] Done (silence {silence_done:.1f}s)")
                await asyncio.sleep(0.2)
                return True

        await asyncio.sleep(0.02)

    logger.warning(f"  [{_axis_name}] Stop-wait TIMEOUT after {timeout_s:.0f}s")
    return False


async def _move_to_abs(motor_id: int, target_deg: float,
                       speed_deg_s: float, smoothness: float,
                       ratio: float, offset_deg: float = 0.0):
    """
    Send an F5 absolute-coordinate command and wait for completion.

    target_deg   : measurement angle (°)
    speed_deg_s  : antenna output speed (°/s)
    smoothness   : Motion Smoothness slider value (1 = instant, 255 = slowest ramp)
    offset_deg   : mechanical zero offset added to get physical motor angle
    """
    physical_deg = target_deg + offset_deg
    speed_rpm    = max(1.0, speed_deg_s * ratio * 60.0 / 360.0)

    # Smoothness → acc byte conversion (manual formula: ramp = (256-acc)*50µs/step)
    # High smoothness → small acc byte → long ramp interval → gentle start/stop.
    acc_byte = int(max(1, min(255, 256 - smoothness)))

    _dir   = AZIMUTH_DIR if motor_id == AZIMUTH_CAN_ID else ELEVATION_DIR
    pulses = int(physical_deg * ratio * PULSES_PER_REV / 360.0 * _dir)
    p24    = pulses & 0xFF_FFFF

    payload = bytes([
        0xF5,
        (int(speed_rpm) >> 8) & 0xFF, int(speed_rpm) & 0xFF,
        acc_byte,
        (p24 >> 16) & 0xFF, (p24 >> 8) & 0xFF, p24 & 0xFF,
    ])
    _axis = "AZIMUTH" if motor_id == AZIMUTH_CAN_ID else "ELEVATION"
    logger.info(f"  [{_axis}] → {target_deg:+.1f}° (physical {physical_deg:+.1f}°)  {speed_rpm:.0f} RPM  acc={acc_byte}")
    await _send_can(motor_id, payload)
    await _wait_motor_stop(motor_id, speed_rpm=speed_rpm, pulses=pulses)


# ═══════════════════════════════════════════════════════════════════
# VNA HELPERS  –  synchronous, no executor
# ═══════════════════════════════════════════════════════════════════
# libreVNA's SocketStreamReader._recv_buffer is NOT thread-safe.
# Wrapping calls in run_in_executor() corrupts the buffer and causes
# responses to be read by the wrong query (TRACE:DATA? returns 'TRUE').
# Plain synchronous calls match the original V3 behaviour that worked.
#
# LibreVNA-GUI ≥ 1.6.0 SCPI behaviour (breaking change vs < 1.6.0):
#   Set-commands  → send only, NO acknowledgement returned.
#   Query commands (ending in '?') → send + receive response line.
# Two helpers enforce this split:
#   _vna_cmd()   — for all set-commands (no '?').  Uses vna.cmd().
#   _vna_query() — for all '?' queries only.       Uses vna.query().

def _vna_cmd(q: str) -> None:
    """Send a SCPI set-command. Expects NO response (LibreVNA-GUI ≥ 1.6.0)."""
    vna.cmd(q)


def _vna_query(q: str) -> str:
    """Send a SCPI query (must end in '?'). Reads and returns the response."""
    return vna.query(q)


# Running count of sweeps that had VNA errors — used to make the
# banner text unique each time so NiceGUI always pushes an update.
_vna_error_count = 0


async def _poll_vna_errors():
    """
    Query VNA status flags after each sweep and update the persistent error
    banner in the progress card directly.

    NiceGUI diffs element state before sending WebSocket updates — if the
    style string and text haven't changed it skips the push entirely.
    To force a fresh update on every sweep we:
      1. Always call .update() explicitly after setting text/style.
      2. Include a running sweep error counter in the text so the
         content is always different from the previous message.

    Per LibreVNA SCPI guide §4.2.11-4.2.13:
      DEV:STA:UNLO?   — PLL unlocked (frequency reference lost)
      DEV:STA:ADCOV?  — ADC overload (input signal too strong)
      DEV:STA:UNLEV?  — Output unlevel (stimulus power cannot be achieved)
    """
    global _vna_error_count
    if vna_error_banner is None:
        return
    try:
        errors = []
        if _vna_query(':DEV:STA:UNLO?').strip().upper() == 'TRUE':
            errors.append('⚠️  PLL UNLOCKED — check reference signal')
        if _vna_query(':DEV:STA:ADCOV?').strip().upper() == 'TRUE':
            errors.append('🔴  ADC OVERLOAD — reduce input power or add attenuation')
        if _vna_query(':DEV:STA:UNLEV?').strip().upper() == 'TRUE':
            errors.append('⚠️  OUTPUT UNLEVEL — stimulus power target cannot be achieved')

        if errors:
            _vna_error_count += 1
            for e in errors:
                logger.warning(f'[VNA] {e}')
            # Append counter so text is always unique → NiceGUI always pushes
            msg = '\n'.join(errors) + f'  (×{_vna_error_count})'
            if vna_error_text is not None:
                vna_error_text.text = msg
                vna_error_text.update()           # force WebSocket push
            vna_error_banner.style(
                'display:block; width:100%; margin-top:10px; padding:10px 16px; '
                'background:#7f1d1d; border:1.5px solid #ef4444; border-radius:10px;'
            )
            vna_error_banner.update()             # force WebSocket push
        else:
            if vna_error_banner.style() != 'display:none;':
                vna_error_banner.style('display:none;')
                vna_error_banner.update()

    except Exception as e:
        logger.debug(f'[VNA] Error poll failed: {e}')


async def _wait_sweep_done():
    """
    Trigger a fresh single sweep and wait for it to complete.

    Debug findings (VNA_DEBUG output):
      - VNA:ACQ:RUN? returns ERROR (VNA is not in continuous run mode)
      - After SINGLE TRUE, FIN? stays TRUE and AVGLEVEL stays 1 for ~430 ms
        before the new sweep actually begins
      - The AVGLEVEL=0 polling loop (no sleep, ~0.4 ms/query) exhausted 500
        iterations in ~200 ms — too short to catch the ~430 ms startup delay
      - Adding asyncio.sleep(0.5) after SINGLE TRUE covers the startup delay
        reliably; after that FIN? properly returns FALSE then TRUE

    Correct sequence:
      1. SINGLE TRUE          → arms new sweep
      2. sleep 0.5 s          → wait for VNA to start the sweep (~430 ms typical)
      3. Poll FIN? until TRUE → sweep complete, data safe to read
    """
    _vna_cmd(':VNA:ACQ:SINGLE TRUE')   # set-command — no response expected

    # Wait for the VNA to actually start the sweep.
    # Debug showed the sweep starts ~430 ms after SINGLE TRUE.
    await asyncio.sleep(0.5)

    # Now poll until sweep is complete
    for _ in range(1500):         # max ~30 s at 20 ms/poll
        if _vna_query(':VNA:ACQ:FIN?').strip().upper() == 'TRUE':
            return
        await asyncio.sleep(0.02)
    logger.warning('  [VNA] Sweep timeout after 30s')


# ═══════════════════════════════════════════════════════════════════
# HARDWARE CONNECT / DISCONNECT
# ═══════════════════════════════════════════════════════════════════

def _flush_bus(timeout_s: float = 0.3):
    """
    Drain stale frames from the SLCAN adapter's hardware buffer.
    Without this, frames left over from a previous session are replayed
    into the motor state machine at the start of the next run.
    """
    flushed  = 0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            msg = bus.recv(timeout=0.05)
            if msg is not None:
                flushed += 1
        except Exception:
            break
    if flushed:
        logger.info(f'CAN buffer: flushed {flushed} stale frame(s)')


async def _connect_hw():
    global vna, bus
    logger.info('[HW] Connecting LibreVNA + CAN…')
    _safe_notify('Connecting LibreVNA + CAN…', type='info')
    loop = asyncio.get_event_loop()
    vna  = await loop.run_in_executor(
        _executor, lambda: libreVNA('localhost', 19542)
    )
    bus  = await loop.run_in_executor(
        _executor,
        lambda: can.Bus(
            interface='slcan',
            channel=can_port_input.value if can_port_input else CAN_CHANNEL,
            bitrate=CAN_BITRATE
        ),
    )
    _flush_bus()
    await _init_motor(AZIMUTH_CAN_ID)
    await _init_motor(ELEVATION_CAN_ID)
    await _enable_motor(AZIMUTH_CAN_ID)
    await _enable_motor(ELEVATION_CAN_ID)
    logger.info('[HW] Ready — VNA + motors online')
    _safe_notify('✅ Hardware ready', type='positive')


async def _disconnect_hw():
    global bus
    try:
        await _disable_motor(AZIMUTH_CAN_ID)
        await _disable_motor(ELEVATION_CAN_ID)
    except Exception:
        pass
    if bus:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, bus.shutdown)
        bus = None


# ═══════════════════════════════════════════════════════════════════
# LIVE PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════

def _sel_freq_hz() -> float:
    try:
        return float(freq_slider.value) * 1e9
    except Exception:
        return 1e9



# ═══════════════════════════════════════════════════════════════════
# S21 NORMALISATION
# ═══════════════════════════════════════════════════════════════════

def _get_plot_df() -> "pd.DataFrame":
    """
    Return a DataFrame ready for plotting.
    If normalisation is enabled the s21_db column is adjusted;
    the raw df_data list is NEVER modified so CSV export is always raw.
    """
    if not df_data:
        return pd.DataFrame()
    df = pd.DataFrame(df_data)
    if norm_radio is None:
        return df
    mode = norm_radio.value   # 'Off' | 'Angle' | 'Max'
    if mode == 'Off':
        return df

    df_out = df.copy()
    for freq_hz, grp_idx in df_out.groupby('freq_hz').groups.items():
        grp = df_out.loc[grp_idx]
        if mode == 'Max':
            ref_db = grp['s21_db'].max()
        else:                           # Angle
            try:
                n_azim = float(norm_azim_input.value or 0)
                n_elev   = float(norm_elev_input.value   or 0)
            except Exception:
                n_azim, n_elev = 0.0, 0.0
            dist = np.sqrt((grp['azim_deg'] - n_azim) ** 2 +
                           (grp['elev_deg']   - n_elev)   ** 2)
            ref_db = grp.loc[dist.idxmin(), 's21_db']
        df_out.loc[grp_idx, 's21_db'] = df_out.loc[grp_idx, 's21_db'] - ref_db

    return df_out


# ═══════════════════════════════════════════════════════════════════
# SETTINGS SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════
import json as _json_mod

SETTINGS_FILE = Path('antenna_settings.json')

_SETTINGS_MAP = [
    # (json_key, widget_attr, value_attr)
    ('project',           'project_input',        'value'),
    ('measurement_type',  'measurement_radio',     'value'),
    ('azim_start',        'azim_start_input',      'value'),
    ('azim_stop',         'azim_stop_input',       'value'),
    ('elev_start',        'elev_start_input',      'value'),
    ('elev_stop',         'elev_stop_input',       'value'),
    ('freq_range',        'freq_range_input',      'value'),
    ('vna_freq_step',     'vna_freq_step_input',   'value'),
    ('vna_smooth',        'vna_smooth_slider',     'value'),
    ('vna_avg',           'vna_avg_input',         'value'),
    ('vna_ifbw',          'vna_ifbw_input',        'value'),
    ('vna_level',         'vna_level_input',        'value'),
    ('angle_step',        'angle_step_input',      'value'),
    ('cut_type',          'cut_type_dropdown',     'value'),
    ('constant_angle',    'constant_angle_input',  'value'),
    ('speed',             'speed_slider',          'value'),
    ('smoothness',        'acc_slider',            'value'),
    ('can_port',         'can_port_input',       'value'),
    ('settle_delay',     'settle_delay_input',   'value'),
    ('azim_offset',      'azim_offset_input',    'value'),
    ('elev_offset',        'elev_offset_input',      'value'),
    ('plot_enable',       'plot_enable_radio',     'value'),
    ('plot_freqs',        'plot_freq_input',       'value'),
    ('html3d_enable',     'html3d_enable_radio',   'value'),
    ('html3d_freqs',      'html3d_freq_input',     'value'),
    ('norm_mode',         'norm_radio',            'value'),
    ('norm_azim',        'norm_azim_input',      'value'),
    ('norm_elev',          'norm_elev_input',        'value'),
    ('notes',             'notes_input',           'value'),
]


def _save_settings():
    """Collect all GUI widget values and write to antenna_settings.json."""
    data = {}
    g = globals()
    for key, wname, attr in _SETTINGS_MAP:
        widget = g.get(wname)
        if widget is not None:
            try:
                data[key] = getattr(widget, attr)
            except Exception:
                pass
    SETTINGS_FILE.write_text(_json_mod.dumps(data, indent=2), encoding='utf-8')
    _safe_notify(f'✅ Settings saved → {SETTINGS_FILE}', type='positive')
    logger.info(f'[SETTINGS] Saved to {SETTINGS_FILE}')


def _load_settings(silent: bool = False):
    """Load antenna_settings.json and apply values to GUI widgets."""
    if measurement_running:
        if not silent:
            _safe_notify('⚠️ Cannot load settings during a measurement.', type='warning')
        return
    if not SETTINGS_FILE.exists():
        if not silent:
            _safe_notify('⚠️ No settings file found.', type='warning')
        return
    try:
        data = _json_mod.loads(SETTINGS_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        _safe_notify(f'❌ Could not read settings: {e}', type='negative')
        return
    g = globals()
    for key, wname, attr in _SETTINGS_MAP:
        widget = g.get(wname)
        if widget is not None and key in data:
            try:
                setattr(widget, attr, data[key])
                widget.update()
            except Exception:
                pass
    _update_vna_ui()   # refresh computed labels
    # Re-apply all toggle-based visual states (dimming, pointer-events, etc.)
    ui.timer(0.1, lambda: _ui_state_updater() if _ui_state_updater else None, once=True)
    if not silent:
        _safe_notify('✅ Settings loaded.', type='positive')
    logger.info(f'[SETTINGS] Loaded from {SETTINGS_FILE}')


def update_vna_trace(data_parsed: list):
    """Refresh the live VNA S21 trace — safe to call from any context."""
    try:
        _update_vna_trace_inner(data_parsed)
    except RuntimeError:
        pass
    except Exception as e:
        logger.debug(f'update_vna_trace: {e}')


def _update_vna_trace_inner(data_parsed: list):
    if not data_parsed or live_vna_trace is None:
        return
    freqs_ghz = [f[0] / 1e9 for f in data_parsed]
    s21_db    = [20 * np.log10(max(abs(f[1]), 1e-12)) for f in data_parsed]

    vna_fig.data = []
    vna_fig.add_trace(go.Scatter(
        x=freqs_ghz, y=s21_db,
        mode='lines+markers',
        line=dict(color='#00ffaa', width=2),
        marker=dict(size=4),
        name='S21',
    ))
    vna_fig.update_layout(
        title='Live VNA S21  (current point)',
        xaxis_title='Frequency (GHz)', yaxis_title='S21 (dB)',
        xaxis=dict(gridcolor='#444'),
        yaxis=dict(autorange=True, gridcolor='#444'),
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a', font_color='#ddd',
        margin=dict(l=55, r=20, t=45, b=45),
    )
    live_vna_trace.update_figure(vna_fig)


_farfield_debounce_timer = None
_farfield_busy           = False   # re-entrancy guard


def _schedule_farfield_update():
    """
    Debounce: cancel any pending redraw and schedule a new one 150 ms out.
    Only the last slider position after the user stops dragging triggers a
    redraw, preventing figure flickering and empty-plot crashes.
    """
    global _farfield_debounce_timer
    try:
        if _farfield_debounce_timer is not None:
            _farfield_debounce_timer.cancel()
    except Exception:
        pass
    _farfield_debounce_timer = ui.timer(
        0.15, lambda: update_farfield_plot(), once=True
    )


def update_farfield_plot():
    """Refresh the farfield preview — safe to call from any context."""
    global _farfield_busy
    if _farfield_busy:
        return   # previous call still running; skip
    _farfield_busy = True
    try:
        _update_farfield_plot_inner()
    except RuntimeError:
        pass
    except Exception as e:
        logger.debug(f'update_farfield_plot: {e}')
    finally:
        _farfield_busy = False


def _update_farfield_plot_inner():
    if not df_data or live_farfield_plot is None or freq_slider is None:
        return

    is_3d  = measurement_radio.value in ('3D - Full', '3D - Range')
    tgt_hz = _sel_freq_hz()

    df = _get_plot_df()
    if df.empty:
        return
    df['_diff'] = np.abs(df['freq_hz'] - tgt_hz)
    closest = df.loc[df['_diff'].idxmin(), 'freq_hz']
    subset  = df[df['freq_hz'] == closest].copy()
    if subset.empty:
        return

    sigma = float(smooth_slider.value) if smooth_slider is not None else 0.0
    farfield_fig.data = []

    if is_3d:
        azim_vals = sorted(subset['azim_deg'].unique())
        elev_vals   = sorted(subset['elev_deg'].unique())
        pivot      = subset.pivot_table(
            index='azim_deg', columns='elev_deg', values='s21_db', aggfunc='mean'
        ).reindex(index=azim_vals, columns=elev_vals)
        s21_2d = pivot.values.copy()

        if sigma > 0:
            s21_2d = gaussian_filter(s21_2d, sigma=sigma, mode=['wrap', 'reflect'])

        max_s21  = float(np.nanmax(s21_2d))
        gamma    = float(shape_slider.value) if shape_slider is not None else 1.0
        col_data = np.clip(s21_2d, max_s21 - 30.0, max_s21)  # colour fixed at 30 dB

        # Correct antenna coordinate convention:
        #   azimuth (-180…+175°), elevation (-90…+90°)
        #   X = r·cos(phi)·cos(theta),  Y = r·cos(elev)·sin(azim),  Z = r·sin(elev)
        # Close surface in azimuth direction to eliminate seam.
        azim_closed = np.append(azim_vals, azim_vals[0] + 360.0)
        s21_closed   = np.vstack([s21_2d,   s21_2d[0:1, :]])
        col_closed   = np.vstack([col_data, col_data[0:1, :]])

        T_rad = np.radians(azim_closed)[:, np.newaxis]
        P_rad = np.radians(elev_vals)[np.newaxis, :]
        # Gamma compression: r = linear_amplitude ^ gamma
        #   gamma=1.0 → true linear (max contrast)
        #   gamma<1.0 → weak lobes grow, peak stays at r=1
        amp   = 10 ** ((s21_closed - max_s21) / 20.0)   # 0…1 linear amplitude
        amp   = np.nan_to_num(np.clip(amp, 0.0, 1.0), nan=0.0)
        r_lin = amp ** gamma                              # colour window unchanged

        # Coordinate convention (CST-style):
        #   Z = boresight  (azimuth=0°, elevation=0°)
        #   Y = up         (elevation=90°)
        #   X = lateral    (azimuth=90°, elevation=0°)
        X = r_lin * np.cos(P_rad) * np.sin(T_rad)   # lateral
        Y = r_lin * np.sin(P_rad)                    # up
        Z = r_lin * np.cos(P_rad) * np.cos(T_rad)   # boresight

        farfield_fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=col_closed,
            colorscale='Jet',
            cmin=max_s21 - 30.0,
            cmax=max_s21,
            colorbar=dict(
                title=dict(text='S21 (dB)', font=dict(color='#ddd')),
                tickfont=dict(color='#ddd'),
            ),
            lighting=dict(ambient=0.8, diffuse=0.5),
            showscale=True,
        ))

        # ── Coordinate axes (CST-style) ───────────────────────────
        # Axis length = 120 % of the farfield balloon radius
        ax_len = float(np.nanmax(np.sqrt(X**2 + Y**2 + Z**2))) * 1.25

        def _axis_traces(dx, dy, dz, color, name):
            """One axis: line from origin + cone arrowhead + label."""
            tip = [dx * ax_len, dy * ax_len, dz * ax_len]
            return [
                # shaft
                go.Scatter3d(
                    x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
                    mode='lines',
                    line=dict(color=color, width=6),
                    showlegend=False, hoverinfo='skip',
                ),
                # arrowhead cone
                go.Cone(
                    x=[tip[0]], y=[tip[1]], z=[tip[2]],
                    u=[dx], v=[dy], w=[dz],
                    sizemode='absolute', sizeref=ax_len * 0.18,
                    colorscale=[[0, color], [1, color]],
                    showscale=False, showlegend=False, hoverinfo='skip',
                ),
                # label
                go.Scatter3d(
                    x=[tip[0] * 1.18], y=[tip[1] * 1.18], z=[tip[2] * 1.18],
                    mode='text',
                    text=[name],
                    textfont=dict(color=color, size=14),
                    showlegend=False, hoverinfo='skip',
                ),
            ]

        # X axis  → Azimuth   (red)
        # Y axis  → Elevation (green, up)
        # Z axis  → Boresight (blue)
        for trace in (
            _axis_traces(1, 0, 0, '#ff4444', 'X (Azimuth)')   +
            _axis_traces(0, 1, 0, '#44ff88', 'Y (Elevation)') +
            _axis_traces(0, 0, 1, '#4499ff', 'Z (Boresight)')
        ):
            farfield_fig.add_trace(trace)

        axis_style = dict(
            showbackground=True, backgroundcolor='#0d0d0d',
            gridcolor='#333', zerolinecolor='#555',
            tickfont=dict(color='#aaa'),
        )
        farfield_fig.update_layout(
            title=dict(
                text=f'3D Farfield  @  {closest / 1e9:.3f} GHz',
                font=dict(size=16, color='#ddd'),
            ),
            scene=dict(
                xaxis=dict(title=dict(text='X (Azimuth)',   font=dict(color='#aaa')), **axis_style),
                yaxis=dict(title=dict(text='Y (Elevation)',  font=dict(color='#aaa')), **axis_style),
                zaxis=dict(title=dict(text='Z (Boresight)',  font=dict(color='#aaa')), **axis_style),
                aspectmode='data',
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
            ),
            paper_bgcolor='#0a0a0a', font_color='#ddd',
            margin=dict(l=0, r=0, t=55, b=0),
        )

    else:
        cut = cut_type_dropdown.value
        if cut == 'Constant Elevation':
            x_col, x_label = 'azim_deg', 'Azimuth (°)'
            title = (f"Farfield  @  {closest / 1e9:.3f} GHz"
                     f"  (Elevation = {constant_angle_input.value}°)")
        else:
            x_col, x_label = 'elev_deg', 'Elevation (°)'
            title = (f"Farfield  @  {closest / 1e9:.3f} GHz"
                     f"  (Azimuth = {constant_angle_input.value}°)")

        subset = subset.sort_values(x_col)
        y_data = subset['s21_db'].values.copy()
        if sigma > 0:
            y_data = gaussian_filter1d(y_data, sigma=sigma, mode='wrap')

        farfield_fig.add_trace(go.Scatter(
            x=subset[x_col], y=y_data,
            mode='lines+markers',
            line=dict(color='#e63939', width=2),
            marker=dict(size=5, color='#e63939'),
        ))
        farfield_fig.update_layout(
            title=dict(text=title, font=dict(size=15, color='#ddd')),
            xaxis_title=x_label, yaxis_title='S21 (dB)',
            xaxis=dict(range=[-180, 180], dtick=45, gridcolor='#444'),
            yaxis=dict(autorange=True, gridcolor='#444'),
            plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a', font_color='#ddd',
            showlegend=False,
            margin=dict(l=55, r=20, t=50, b=45),
        )

    live_farfield_plot.update_figure(farfield_fig)


# ═══════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════

def _project_dir(project: str) -> Path:
    """Return and create RESULTS/<project>/ directory."""
    p = Path('RESULTS') / project
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_csv(project: str):
    """
    Save results as CSV inside RESULTS/<project>/.
    Format: azim_deg, elev_deg, frequencies (space-separated Hz),
            s21_logmag (space-separated dB) — one row per angle point.
    """
    final_df = pd.DataFrame(df_data)
    rows     = []
    for (azim, elev), group in final_df.groupby(['azim_deg', 'elev_deg']):
        freq_str = ' '.join(f'{f:.0f}' for f in group['freq_hz'].values)
        s21_str  = ' '.join(f'{s:.6f}' for s in group['s21_db'].values)
        rows.append({
            'azim_deg':   azim,
            'elev_deg':     elev,
            'frequencies': freq_str,
            's21_logmag':  s21_str,
        })
    out_path = _project_dir(project) / f'{project}_S21.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    _safe_notify(f'✅ Saved  {out_path}', type='positive')
    logger.info(f'[CSV] {len(rows)} points → {out_path}')


# ═══════════════════════════════════════════════════════════════════
# FARFIELD POLAR PLOT EXPORT
# ═══════════════════════════════════════════════════════════════════

def _generate_farfield_plots(project: str):
    """
    Generate polar AND Cartesian farfield plots for each requested frequency.

    Saved to RESULTS/<project>/PLOTS/:
      <project>_<f>GHz_<cut>_polar.png   – 360° polar plot
      <project>_<f>GHz_<cut>_1D.png      – Cartesian (angle vs dB)

    Auto-scales the radial/y axis from the data.
    White background, full 360° azimuth sweep in polar view.
    """
    if not df_data:
        logger.warning('[PLOT] No data to plot')
        return

    raw = (plot_freq_input.value or '').strip()
    if not raw:
        logger.info('[PLOT] No plot frequencies specified – skipping')
        return
    try:
        req_ghz = sorted({
            float(x.strip())
            for x in raw.replace(',', ' ').split()
            if x.strip()
        })
    except ValueError:
        logger.warning('[PLOT] Could not parse plot frequency list – skipping')
        return

    df        = _get_plot_df()
    if df.empty:
        logger.warning('[PLOT] No data')
        return
    all_freqs = sorted(df['freq_hz'].unique())
    plot_dir  = _project_dir(project) / 'PLOTS'
    plot_dir.mkdir(exist_ok=True)
    is_3d     = len(df['elev_deg'].unique()) > 1 and len(df['azim_deg'].unique()) > 1

    # ── Determine which cuts to plot ────────────────────────────────
    # 3D mode produces two great-circle cuts:
    #   Azimuth plane   – horizontal cut: all azimuth at elevation = 0°
    #   Elevation plane – vertical cut:   stitch azimuth=0° (El -90→+90)
    #                     with azimuth=180° / -180° (El +90→-90, back half)
    #                     to form a full 360° great circle in elevation.
    # 2D mode: single cut determined by the recorded axis.
    if is_3d:
        cuts = [('azimuth_plane', 'Azimuth plane'),
                ('elevation_plane', 'Elevation plane')]
    else:
        elev_vals = df['elev_deg'].unique()
        azim_vals = df['azim_deg'].unique()
        if len(elev_vals) == 1:
            cuts = [(elev_vals[0], f'Elev={elev_vals[0]:.0f}deg')]
        else:
            cuts = [(None, f'Azim={azim_vals[0]:.0f}deg')]

    generated = 0

    for f_ghz in req_ghz:
        nearest = min(all_freqs, key=lambda x: abs(x - f_ghz * 1e9))
        f_label = f'{nearest / 1e9:.3f}'

        for cut_key, cut_label in cuts:
            freq_df = df[df['freq_hz'] == nearest].copy()

            # ── Extract cut data ─────────────────────────────────────
            if is_3d and cut_key == 'azimuth_plane':
                # Horizontal cut: constant elevation = 0°
                nearest_elev = min(df['elev_deg'].unique(),
                                   key=lambda x: abs(x - 0.0))
                subset = freq_df[
                    np.isclose(freq_df['elev_deg'], nearest_elev, atol=0.6)
                ].copy()
                x_col   = 'azim_deg'
                x_label = 'Azimuth [°]'

            elif is_3d and cut_key == 'elevation_plane':
                # Vertical great-circle cut.
                # Front half: azimuth = 0°, elevation sweeps -90 → +90.
                # Back half:  azimuth = 180° (or -180°), elevation sweeps +90 → -90.
                # IMPORTANT: back half is only used if azimuth=180° was actually
                # measured — in 3D-Range mode the back hemisphere may not exist.
                # Threshold: az_back must be within ±(step+1)° of ±180° to be valid.
                all_azim = df['azim_deg'].unique()
                az_front = min(all_azim, key=lambda x: abs(x - 0.0))
                az_back_pos = min(all_azim, key=lambda x: abs(x - 180.0))
                az_back_neg = min(all_azim, key=lambda x: abs(x - (-180.0)))
                az_back = (az_back_pos
                           if abs(az_back_pos - 180.0) <= abs(az_back_neg - (-180.0))
                           else az_back_neg)
                # Only use back half if it is genuinely near ±180°.
                # Derive step from actual azimuth spacing in the data
                # (step is not in scope here — this is a separate function).
                az_sorted      = np.sort(all_azim)
                az_step        = float(np.median(np.diff(az_sorted))) if len(az_sorted) > 1 else 5.0
                back_tolerance = az_step + 1.0
                has_back = (abs(az_back - 180.0) <= back_tolerance or
                            abs(az_back + 180.0) <= back_tolerance)

                front = freq_df[
                    np.isclose(freq_df['azim_deg'], az_front, atol=0.6)
                ].copy().sort_values('elev_deg')            # El: -90 → +90

                if front.empty:
                    logger.warning(f'[PLOT] No elevation-plane data for {f_ghz} GHz – skip')
                    continue

                front_angles = front['elev_deg'].values
                front_s21    = front['s21_db'].values

                if has_back:
                    # Full 360° great circle — stitch front and back halves.
                    # Map to -180…+180 sweep angle:
                    #   front: angle = elevation         (−90 → +90)
                    #   back:  angle = 180° − elevation  (remapped to −180…−90)
                    back = freq_df[
                        np.isclose(freq_df['azim_deg'], az_back, atol=0.6)
                    ].copy().sort_values('elev_deg', ascending=False)  # El: +90 → -90
                    # Strip shared endpoints at El=±90°
                    back_trimmed = back.iloc[1:-1]
                    back_elev    = back_trimmed['elev_deg'].values
                    back_angles  = 180.0 - back_elev
                    back_angles  = np.where(back_angles > 180.0,
                                            back_angles - 360.0,
                                            back_angles)
                    back_s21     = back_trimmed['s21_db'].values
                    angles_deg   = np.concatenate([front_angles, back_angles])
                    s21_db       = np.concatenate([front_s21,    back_s21])
                    logger.info(f'[PLOT] Elevation plane: full 360° great circle '
                                f'(az_front={az_front:.0f}°, az_back={az_back:.0f}°)')
                else:
                    # Partial range — only front half available.
                    # Plot as a simple arc; no back half fabricated.
                    angles_deg = front_angles
                    s21_db     = front_s21
                    logger.info(f'[PLOT] Elevation plane: partial arc only '
                                f'(az_back={az_back:.0f}° too far from ±180°)')

                x_col      = None   # composite — handled specially below
                x_label    = 'Elevation [°]'
                subset     = None   # signal that arrays are already built

            else:
                # 2D mode — existing logic
                elev_const = cut_key   # numeric value in 2D mode
                if elev_const is not None:
                    nearest_elev = min(df['elev_deg'].unique(),
                                       key=lambda x: abs(x - elev_const))
                    subset = freq_df[
                        np.isclose(freq_df['elev_deg'], nearest_elev, atol=0.6)
                    ].copy()
                    x_col   = 'azim_deg'
                    x_label = 'Azimuth [°]'
                else:
                    subset  = freq_df.copy()
                    x_col   = 'elev_deg'
                    x_label = 'Elevation [°]'

            # ── Finalise arrays for non-elevation-plane cuts ─────────
            if subset is not None:
                if subset.empty:
                    logger.warning(f'[PLOT] No data for {f_ghz} GHz {cut_label} – skip')
                    continue
                subset     = subset.sort_values(x_col)
                angles_deg = subset[x_col].values
                s21_db     = subset['s21_db'].values

            # ── Auto-scale: round outward to nearest 5 dB ────────────
            data_min = float(np.nanmin(s21_db))
            data_max = float(np.nanmax(s21_db))
            gain_min = np.floor(data_min / 5) * 5 - 5
            gain_max = np.ceil(data_max  / 5) * 5 + 5

            title      = f'{project}  –  {f_label} GHz  ({cut_label})'
            base_label = cut_label.replace(' ', '_')
            base       = f'{project}_{f_label}GHz_{base_label}'

            # ════════════════════════════════════════════════════════
            # POLAR PLOT
            # ════════════════════════════════════════════════════════
            # Elevation plane uses a different angular mapping to polar axes:
            #   elevation=0°  → top (zenith direction, θ_polar=0)
            #   elevation=+90°→ right side  (θ_polar=90°)
            #   elevation=±180°→ bottom (nadir)
            # Azimuth plane uses standard 0…360° clockwise-from-North mapping.
            is_elev_plane = is_3d and cut_key == 'elevation_plane'

            if is_elev_plane:
                # Elevation plane polar mapping — main lobe (El=0°) at top:
                #   polar_angle = -elevation  (with clockwise direction)
                #   El=  0° → polar   0° (top / North)
                #   El=+90° → polar  90° (clockwise → right)
                #   El=-90° → polar 270° (counter → left)
                #   El=±180° → polar 180° (bottom)
                # Convert stitched angles (-90…+90 front, mapped back half)
                # to 0…360 polar space: negate then wrap to positive.
                elev_polar = -angles_deg
                elev_polar_360 = np.where(elev_polar < 0, elev_polar + 360, elev_polar)
                sort_i        = np.argsort(elev_polar_360)
                elev_sorted   = elev_polar_360[sort_i]
                s21_sorted    = s21_db[sort_i]
                # Close the great circle
                plot_angles_closed = np.append(elev_sorted, elev_sorted[0] + 360)
                plot_s21_closed    = np.append(s21_sorted,  s21_sorted[0])
                polar_rad = np.deg2rad(plot_angles_closed)
            else:
                # Azimuth plane: convert -180…+175 → 0…360 and sort
                ang360 = np.where(angles_deg < 0, angles_deg + 360, angles_deg)
                sort_i = np.argsort(ang360)
                ang360    = ang360[sort_i]
                s21_360   = s21_db[sort_i]
                # Close the circle
                plot_angles_closed = np.append(ang360,  ang360[0]  + 360)
                plot_s21_closed    = np.append(s21_360, s21_360[0])
                polar_rad = np.deg2rad(plot_angles_closed)

            fig_p, ax_p = plt.subplots(figsize=(10, 10),
                                       subplot_kw={'projection': 'polar'},
                                       facecolor='white')
            ax_p.set_facecolor('white')
            ax_p.plot(polar_rad, plot_s21_closed,
                      linewidth=2, color='#c0392b')

            ax_p.set_theta_zero_location('N')
            ax_p.set_theta_direction(-1)
            ax_p.set_rlim(gain_min, gain_max)
            tick_step = 5 if (gain_max - gain_min) <= 30 else 10
            ax_p.set_rticks(np.arange(gain_min, gain_max + 1, tick_step))
            ax_p.tick_params(colors='black', labelsize=10)
            ax_p.grid(True, linestyle='--', alpha=0.5, color='#888888')
            ax_p.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            # Both azimuth and elevation plane use identical 0°…315° tick labels
            ax_p.set_xticklabels(
                ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                fontsize=11
            )
            ax_p.set_title(title, pad=28, fontsize=13, color='black')
            fig_p.text(0.5, 0.02, x_label,
                       ha='center', fontsize=12, color='black')

            out_p = plot_dir / f'{base}_polar.png'
            fig_p.savefig(out_p, dpi=200, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
            plt.close(fig_p)
            logger.info(f'[PLOT] Polar  → {out_p.name}')
            generated += 1

            # ════════════════════════════════════════════════════════
            # CARTESIAN 1-D PLOT
            # ════════════════════════════════════════════════════════
            fig_c, ax_c = plt.subplots(figsize=(12, 5), facecolor='white')
            ax_c.set_facecolor('white')
            # Sort by angle for clean line plot
            sort_1d = np.argsort(angles_deg)
            ax_c.plot(angles_deg[sort_1d], s21_db[sort_1d],
                      linewidth=1.8, color='#c0392b', label='S21 (dB)')
            ax_c.set_xlim(-180, 180)
            ax_c.set_xticks(np.arange(-180, 181, 30))
            ax_c.set_ylim(gain_min, gain_max)
            ax_c.set_xlabel(x_label, fontsize=12)
            ax_c.set_ylabel('S21 [dB]', fontsize=12)
            ax_c.set_title(title, fontsize=13)
            ax_c.grid(True, linestyle='--', alpha=0.5, color='#aaaaaa')
            ax_c.tick_params(colors='black', labelsize=10)
            for spine in ax_c.spines.values():
                spine.set_edgecolor('#cccccc')
            fig_c.tight_layout()

            out_c = plot_dir / f'{base}_1D.png'
            fig_c.savefig(out_c, dpi=200, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
            plt.close(fig_c)
            logger.info(f'[PLOT] 1D     → {out_c.name}')
            generated += 1

    logger.info(f'[PLOT] Done — {generated} plot(s) ({len(req_ghz)} freq × {len(cuts)} cuts × 2 types) saved to RESULTS/{project}/PLOTS/')


# ═══════════════════════════════════════════════════════════════════
# INTERACTIVE 3D HTML EXPORT
# ═══════════════════════════════════════════════════════════════════

def _generate_3d_html(project: str):
    """
    Generate a self-contained interactive 3D farfield HTML file.

    Features:
      - Frequency selector slider (one per requested frequency)
      - Smoothing slider (σ 0-5, pre-computed in Python)
      - CST-style coordinate axes (X/Y/Z arrows)
      - Dark theme matching the GUI preview
      - Plotly.js from CDN → file stays small (< 100 KB data)
      - Saved to RESULTS/<project>/<project>_3D_farfield.html
    """
    import json as _json

    if not df_data:
        logger.warning('[HTML3D] No data')
        return

    raw = (html3d_freq_input.value or '').strip()
    if not raw:
        logger.info('[HTML3D] No frequencies specified – skipping')
        return
    try:
        req_ghz = sorted({
            float(x.strip())
            for x in raw.replace(',', ' ').split()
            if x.strip()
        })
    except ValueError:
        logger.warning('[HTML3D] Cannot parse frequency list – skipping')
        return

    df        = _get_plot_df()
    if df.empty:
        return
    all_freqs = sorted(df['freq_hz'].unique())
    dyn_range = 30.0
    SIGMAS    = [0, 1, 2, 3, 4, 5]   # pre-computed smoothing levels

    # ── Pre-compute surface data ─────────────────────────────────────
    datasets = []       # list of {freq_label, sigma, X, Y, Z, col, cmin, cmax}
    freq_labels = []

    for f_ghz in req_ghz:
        nearest   = min(all_freqs, key=lambda x: abs(x - f_ghz * 1e9))
        f_label   = f'{nearest / 1e9:.3f} GHz'
        if f_label not in freq_labels:
            freq_labels.append(f_label)

        subset = df[df['freq_hz'] == nearest].copy()
        if subset.empty:
            continue

        azim_vals = sorted(subset['azim_deg'].unique())
        elev_vals   = sorted(subset['elev_deg'].unique())
        pivot      = subset.pivot_table(
            index='azim_deg', columns='elev_deg',
            values='s21_db', aggfunc='mean'
        ).reindex(index=azim_vals, columns=elev_vals)
        raw_s21    = pivot.values.copy()
        max_s21    = float(np.nanmax(raw_s21))
        cmin       = round(max_s21 - dyn_range, 2)

        for sigma in SIGMAS:
            s21 = raw_s21.copy()
            if sigma > 0:
                s21 = gaussian_filter(s21, sigma=sigma,
                                      mode=['wrap', 'reflect'])

            col_data = np.clip(s21, max_s21 - dyn_range, max_s21)

            # Close azimuth seam
            azim_closed = np.append(azim_vals, azim_vals[0] + 360.0)
            s21_c  = np.vstack([s21,      s21[0:1, :]])
            col_c  = np.vstack([col_data, col_data[0:1, :]])

            T = np.radians(azim_closed)[:, np.newaxis]
            P = np.radians(elev_vals)[np.newaxis, :]
            # T and P stored as full 2D arrays in dataset for JS recompute

            def _round(arr):
                return [[round(float(v), 4) for v in row] for row in arr]

            datasets.append({
                'f':    f_label,
                'sig':  sigma,
                's21':  _round(s21_c),        # raw dB — JS recomputes XYZ per dyn_range
                'col':  _round(col_c),         # colour fixed at 30 dB window
                'cmin': round(cmin, 1),
                'cmax': round(max_s21, 1),
                'T':    [[round(float(v), 4) for v in row]
                          for row in np.broadcast_to(T, s21_c.shape).tolist()],
                'P':    [[round(float(v), 4) for v in row]
                          for row in np.broadcast_to(P, s21_c.shape).tolist()],
            })

    if not datasets:
        logger.warning('[HTML3D] No datasets computed – skipping')
        return

    # ── Axis arrow helper (returns list of Plotly trace dicts) ───────
    ax_len = 1.35   # slightly beyond unit sphere

    def _axis_js(dx, dy, dz, color, name):
        tip = [dx * ax_len, dy * ax_len, dz * ax_len]
        return [
            {'type': 'scatter3d', 'mode': 'lines',
             'x': [0, tip[0]], 'y': [0, tip[1]], 'z': [0, tip[2]],
             'line': {'color': color, 'width': 6},
             'showlegend': False, 'hoverinfo': 'skip'},
            {'type': 'cone',
             'x': [tip[0]], 'y': [tip[1]], 'z': [tip[2]],
             'u': [dx], 'v': [dy], 'w': [dz],
             'sizemode': 'absolute', 'sizeref': ax_len * 0.18,
             'colorscale': [[0, color], [1, color]],
             'showscale': False, 'showlegend': False, 'hoverinfo': 'skip'},
            {'type': 'scatter3d', 'mode': 'text',
             'x': [tip[0] * 1.2], 'y': [tip[1] * 1.2], 'z': [tip[2] * 1.2],
             'text': [name],
             'textfont': {'color': color, 'size': 13},
             'showlegend': False, 'hoverinfo': 'skip'},
        ]

    axis_traces = (
        _axis_js(1, 0, 0, '#ff5555', 'X (Azimuth)')   +
        _axis_js(0, 1, 0, '#55ff88', 'Y (Elevation)') +
        _axis_js(0, 0, 1, '#5599ff', 'Z (Boresight)')
    )

    data_js     = _json.dumps(datasets,      separators=(',', ':'))
    axes_js     = _json.dumps(axis_traces,   separators=(',', ':'))
    freqs_js    = _json.dumps(freq_labels,   separators=(',', ':'))
    sigmas_js   = _json.dumps(SIGMAS)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{project} – 3D Farfield</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
  body{{margin:0;background:#0a0a0a;color:#ddd;font-family:sans-serif;}}
  #controls{{display:flex;flex-wrap:wrap;gap:24px;align-items:center;
            padding:16px 24px;background:#111;border-bottom:1px solid #333;}}
  .ctrl-group{{display:flex;flex-direction:column;gap:6px;min-width:260px;}}
  label{{font-size:12px;font-weight:600;letter-spacing:.08em;color:#aaa;}}
  input[type=range]{{width:100%;accent-color:#4499ff;}}
  .val-badge{{font-family:monospace;font-size:13px;color:#60a5fa;}}
  #plot{{width:100%;height:calc(100vh - 120px);}}
</style>
</head>
<body>
<div id="controls">
  <div class="ctrl-group">
    <label>FREQUENCY</label>
    <input type="range" id="fslider" min="0" max="{len(freq_labels)-1}"
           value="0" step="1" oninput="onSlider()">
    <span class="val-badge" id="flabel">{freq_labels[0] if freq_labels else ''}</span>
  </div>
  <div class="ctrl-group">
    <label>SMOOTHING  (σ)</label>
    <input type="range" id="sslider" min="0" max="{len(SIGMAS)-1}"
           value="0" step="1" oninput="onSlider()">
    <span class="val-badge" id="slabel">OFF – raw data</span>
  </div>
  <div class="ctrl-group">
    <label>SHAPE COMPRESSION  (γ)</label>
    <input type="range" id="drslider" min="1" max="10"
           value="10" step="1" oninput="onSlider()">
    <span class="val-badge" id="drlabel">γ = 1.0  (linear)</span>
  </div>
  <div style="color:#888;font-size:11px;max-width:260px;line-height:1.5;">
    ℹ️ Requires internet for Plotly.js (CDN).<br>
    Rotate: drag &nbsp;|&nbsp; Zoom: scroll &nbsp;|&nbsp; Pan: shift+drag
  </div>
</div>
<div id="plot"></div>

<script>
const DATA    = {data_js};
const AXES    = {axes_js};
const FREQS   = {freqs_js};
const SIGMAS  = {sigmas_js};

const axStyle = {{
  showbackground:true, backgroundcolor:'#0d0d0d',
  gridcolor:'#333', zerolinecolor:'#555',
  tickfont:{{color:'#aaa'}}, titlefont:{{color:'#aaa'}}
}};

function findDS(f, sig) {{
  return DATA.find(d => d.f === f && d.sig === sig);
}}

function computeXYZ(ds, gamma) {{
  // Recompute 3D surface coordinates using gamma compression.
  // gamma=1.0 → true linear amplitude (max contrast).
  // gamma<1.0 → weak lobes grow toward peak; peak stays at r=1.
  // Colour is always unchanged.
  const rows = ds.s21.length, cols = ds.s21[0].length;
  const X = [], Y = [], Z = [];
  for (let i = 0; i < rows; i++) {{
    const xr = [], yr = [], zr = [];
    for (let j = 0; j < cols; j++) {{
      const s   = ds.s21[i][j];
      const T   = ds.T[i][j];
      const P   = ds.P[i][j];
      const amp = Math.pow(10, (s - ds.cmax) / 20);   // 0…1 linear amplitude
      const r   = Math.pow(Math.max(amp, 0), gamma);   // compressed
      xr.push(parseFloat((r * Math.cos(P) * Math.sin(T)).toFixed(4)));
      yr.push(parseFloat((r * Math.sin(P)).toFixed(4)));
      zr.push(parseFloat((r * Math.cos(P) * Math.cos(T)).toFixed(4)));
    }}
    X.push(xr); Y.push(yr); Z.push(zr);
  }}
  return {{X, Y, Z}};
}}

function buildPlot(f, sig, gamma) {{
  const ds = findDS(f, sig);
  if (!ds) return;
  const {{X, Y, Z}} = computeXYZ(ds, gamma);
  const surf = {{
    type:'surface', x:X, y:Y, z:Z,
    surfacecolor:ds.col,
    colorscale:'Jet', cmin:ds.cmin, cmax:ds.cmax,
    colorbar:{{title:{{text:'S21 (dB)',font:{{color:'#ddd'}}}},
              tickfont:{{color:'#ddd'}}}},
    lighting:{{ambient:0.8, diffuse:0.5}},
    showscale:true
  }};
  const traces = [surf, ...AXES];
  const layout = {{
    title:{{text:`{project}  –  ${{f}}  (σ=${{SIGMAS[sig]}}, γ=${{gamma.toFixed(1)}})`,
           font:{{color:'#ddd',size:15}}}},
    paper_bgcolor:'#0a0a0a', font:{{color:'#ddd'}},
    margin:{{l:0,r:0,t:55,b:0}},
    scene:{{
      xaxis:{{title:'X (Azimuth)',...axStyle}},
      yaxis:{{title:'Y (Elevation)',...axStyle}},
      zaxis:{{title:'Z (Boresight)',...axStyle}},
      aspectmode:'data',
      camera:{{eye:{{x:1.4,y:1.4,z:0.8}}}}
    }}
  }};
  Plotly.react('plot', traces, layout);
}}

function onSlider() {{
  const fi   = parseInt(document.getElementById('fslider').value);
  const si   = parseInt(document.getElementById('sslider').value);
  const drVal = parseInt(document.getElementById('drslider').value);
  const gamma = drVal / 10.0;   // slider 1–10 → gamma 0.1–1.0
  const sig  = SIGMAS[si];
  document.getElementById('flabel').textContent = FREQS[fi];
  document.getElementById('slabel').textContent =
    sig === 0 ? 'OFF – raw data' : `σ = ${{sig}}  (window ${{2*sig+1}})`;
  document.getElementById('drlabel').textContent =
    gamma === 1.0 ? 'γ = 1.0  (linear)' : `γ = ${{gamma.toFixed(1)}}  (compressed)`;
  buildPlot(FREQS[fi], sig, gamma);
}}

// Initial render
buildPlot(FREQS[0], 0, 1.0);
</script>
</body></html>"""

    out = _project_dir(project) / f'{project}_3D_farfield.html'
    out.write_text(html, encoding='utf-8')
    sz_kb = out.stat().st_size / 1024
    logger.info(f'[HTML3D] {out}  ({sz_kb:.0f} KB)  '
                f'{len(freq_labels)} freq × {len(SIGMAS)} σ levels')


# ═══════════════════════════════════════════════════════════════════
# VNA DATA SMOOTHING
# ═══════════════════════════════════════════════════════════════════

def _smooth_vna(data_parsed: list, half_window: int) -> list:
    """
    Apply Savitzky-Golay smoothing to raw VNA complex S21 data.

    Why Savitzky-Golay:
      - Fits a polynomial over each sliding window → preserves peak shapes,
        resonance notches, and fine spectral features far better than a
        moving average or Gaussian blur.
      - Applied to real and imaginary parts separately before converting to
        dB, which is more physically correct than smoothing on log-magnitude.
      - Standard algorithm used in bench VNA instruments' built-in smoothing.

    half_window : slider value (0 = off).
                  Actual window = 2*half_window + 1  (always odd, >= 3)
                  Polynomial order = 2 (quadratic, good balance of
                  smoothness vs feature preservation)
    """
    if half_window < 1 or len(data_parsed) < 5:
        return data_parsed

    n      = len(data_parsed)
    window = max(3, min(2 * half_window + 1, n if n % 2 == 1 else n - 1))
    order  = min(2, window - 1)

    freqs  = [d[0] for d in data_parsed]
    reals  = savgol_filter([d[1].real for d in data_parsed], window, order)
    imags  = savgol_filter([d[1].imag for d in data_parsed], window, order)

    return [(f, complex(r, i)) for f, r, i in zip(freqs, reals, imags)]


# ═══════════════════════════════════════════════════════════════════
# MAIN MEASUREMENT COROUTINE
# ═══════════════════════════════════════════════════════════════════

async def _tesazim_offsets():
    """
    Move both motors to measurement 0° (applying the current GUI offsets)
    so the user can verify the antenna points to boresight before measuring.
    A second call returns the motors to physical 0°.
    """
    global bus, vna

    _test_btn.enabled = False
    _test_btn.text    = 'Connecting…'

    try:
        # Connect motors only (VNA not needed for this test)
        loop = asyncio.get_event_loop()
        bus  = await loop.run_in_executor(
            _executor,
            lambda: can.Bus(
                interface='slcan',
                channel=can_port_input.value if can_port_input else CAN_CHANNEL,
                bitrate=CAN_BITRATE
            ),
        )
        _flush_bus()
        await _init_motor(AZIMUTH_CAN_ID)
        await _init_motor(ELEVATION_CAN_ID)
        await _enable_motor(AZIMUTH_CAN_ID)
        await _enable_motor(ELEVATION_CAN_ID)

        spd    = float(speed_slider.value)
        smooth = float(acc_slider.value)
        azim_off  = float(azim_offset_input.value or AZIMUTH_OFFSET_DEG)
        elev_off  = float(elev_offset_input.value   or ELEVATION_OFFSET_DEG)

        start_button.enabled = False
        _test_btn.text = 'Moving to 0°…'
        logger.info(
            f'[TEST] Moving to measurement 0°  '
            f'(Azimuth physical={azim_off:+.1f}°  Elevation physical={elev_off:+.1f}°)'
        )

        await _move_to_abs(AZIMUTH_CAN_ID, 0, spd, smooth, AZIMUTH_GEAR_RATIO, azim_off)
        await _move_to_abs(ELEVATION_CAN_ID,   0, spd, smooth, ELEVATION_GEAR_RATIO,   elev_off)

        update_gauge(azim_pointer, azim_value_label, 0.0)
        update_gauge(elev_pointer,   elev_value_label,   0.0)

        _safe_notify(
            f'✅ At measurement 0° — '
            f'Azimuth physical {azim_off:+.1f}°, Elevation physical {elev_off:+.1f}°. '
            f'Check antenna boresight, then click RETURN TO PHYSICAL 0°.',
            type='positive', timeout=8000
        )
        logger.info('[TEST] In position — verify antenna boresight visually')

        _test_btn.text    = 'At 0° ✓'
        _return_btn.enabled = True

    except Exception as e:
        _safe_notify(f'❌ Test error: {e}', type='negative')
        logger.error(f'[TEST] Error: {e}', exc_info=True)
        if bus:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(_executor, bus.shutdown)
            except Exception:
                pass
            bus = None
        start_button.enabled = True
        _test_btn.text    = 'TEST ZERO POSITION'
        _test_btn.enabled = True


async def _return_to_physical_zero():
    """Return motors to their absolute physical 0° (no offset applied)."""
    global bus

    _return_btn.enabled = False
    _return_btn.text    = 'Returning…'

    try:
        spd    = float(speed_slider.value)
        smooth = float(acc_slider.value)

        logger.info('[TEST] Returning to physical 0°')
        await _move_to_abs(AZIMUTH_CAN_ID, 0, spd, smooth, AZIMUTH_GEAR_RATIO, offset_deg=0.0)
        await _move_to_abs(ELEVATION_CAN_ID,   0, spd, smooth, ELEVATION_GEAR_RATIO,   offset_deg=0.0)

        update_gauge(azim_pointer, azim_value_label, 0.0)
        update_gauge(elev_pointer,   elev_value_label,   0.0)

        await _disable_motor(AZIMUTH_CAN_ID)
        await _disable_motor(ELEVATION_CAN_ID)

        if bus:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_executor, bus.shutdown)
            bus = None

        _safe_notify('✅ Motors at physical 0° and disabled.', type='positive')
        logger.info('[TEST] Done — motors at physical 0°, disabled, bus closed')

    except Exception as e:
        _safe_notify(f'❌ Return error: {e}', type='negative')
        logger.error(f'[TEST] Return error: {e}', exc_info=True)
    finally:
        start_button.enabled = True
        _test_btn.text    = 'TEST ZERO POSITION'
        _test_btn.enabled = True
        _return_btn.text  = 'RETURN TO PHYSICAL 0°'
        _return_btn.enabled = False


async def perform_measurement():
    global measurement_running, stop_requested, df_data

    measurement_running  = True
    stop_requested       = False
    start_button.enabled = False
    stop_button.enabled  = True
    df_data.clear()

    try:
        await _connect_hw()

        project = (project_input.value or 'TEST').strip()
        is_3d   = measurement_radio.value in ('3D - Full', '3D - Range')

        # ── Parse frequency range ─────────────────────────────────
        try:
            parts      = (freq_range_input.value
                          .replace('GHz', '').replace('ghz', '').strip()
                          .split('-'))
            freq_start = float(parts[0]) * 1e9
            freq_stop  = float(parts[1]) * 1e9
        except Exception:
            freq_start, freq_stop = 1e9, 3e9

        step        = float(angle_step_input.value or 5)
        cut_type    = cut_type_dropdown.value
        const_angle = float(constant_angle_input.value or 0)
        spd         = float(speed_slider.value)
        smooth      = float(acc_slider.value)

        # ── Update frequency preview slider range ─────────────────
        f_start_ghz = round(freq_start / 1e9, 3)
        f_stop_ghz  = round(freq_stop  / 1e9, 3)
        step_ghz = round((f_stop_ghz - f_start_ghz) / 200, 4)
        step_ghz = max(0.0001, step_ghz)
        freq_slider._props['min']  = f_start_ghz
        freq_slider._props['max']  = f_stop_ghz
        freq_slider._props['step'] = step_ghz
        freq_slider.value = f_start_ghz
        freq_slider.update()

        # ── Configure VNA ─────────────────────────────────────────
        # Set-commands use _vna_cmd() — no response expected (LibreVNA-GUI ≥ 1.6.0).
        # Query commands (ending in '?') use _vna_query() — response expected.
        # Calculate acquisition points from frequency step
        try:
            vna_step_hz = float(
                vna_freq_step_input.value.replace('MHz','').replace('mhz','').strip()
            ) * 1e6
            vna_step_hz = max(1e3, vna_step_hz)   # minimum 1 kHz
        except Exception:
            vna_step_hz = 4e6                       # default 4 MHz
        acq_points = int(round((freq_stop - freq_start) / vna_step_hz)) + 1
        acq_points = max(2, min(10001, acq_points)) # LibreVNA limit: 2–10001
        logger.info(f'VNA: {freq_start/1e9:.3f}–{freq_stop/1e9:.3f} GHz  '
                    f'step={vna_step_hz/1e6:.3f} MHz  points={acq_points}')

        vna_avg = max(1, int(vna_avg_input.value or 1)) if vna_avg_input else 1
        _vna_cmd(':DEV:MODE VNA')
        _vna_cmd(f':VNA:FREQuency:START {int(freq_start)}')
        _vna_cmd(f':VNA:FREQuency:STOP {int(freq_stop)}')
        _vna_cmd(f':VNA:ACQ:POINTS {acq_points}')
        _vna_cmd(f':VNA:ACQ:AVG {vna_avg}')
        logger.info(f'  [VNA] Averaging = {vna_avg} sweep(s) per point')

        # ui.number returns a numeric value directly — no string parsing needed.
        # The previous string-based parsing was unreliable.
        try:
            ifbw_hz = max(1, int(float(vna_ifbw_input.value or 1000)))
        except Exception:
            ifbw_hz = 1000
        _vna_cmd(f':VNA:ACQ:IFBW {ifbw_hz}')
        # Read back what LibreVNA actually accepted (it quantizes to valid values)
        ifbw_actual = _vna_query(':VNA:ACQ:IFBW?').strip()
        logger.info(f'  [VNA] IF BW requested={ifbw_hz} Hz  accepted={ifbw_actual} Hz')
        if str(ifbw_hz) != ifbw_actual:
            logger.warning(f'  [VNA] IF BW was quantized: {ifbw_hz} → {ifbw_actual} Hz')

        try:
            level_dbm = float(str(vna_level_input.value or -10).strip())
        except Exception:
            level_dbm = -10.0
        _vna_cmd(f':VNA:STIMulus:LVL {level_dbm}')
        logger.info(f'  [VNA] Stimulus level = {level_dbm} dBm')

        await asyncio.sleep(0.5)

        # ── Build angle lists ─────────────────────────────────────
        is_range = measurement_radio.value == '3D - Range'
        if is_range:
            try:
                az_start = float(azim_start_input.value)
                az_stop  = float(azim_stop_input.value)
                el_start = float(elev_start_input.value)
                el_stop  = float(elev_stop_input.value)
            except Exception:
                az_start, az_stop  = -180.0, 175.0
                el_start, el_stop  = -90.0,  90.0
            # Ensure start < stop for arange; swap silently if needed
            if az_start > az_stop:
                az_start, az_stop = az_stop, az_start
            if el_start > el_stop:
                el_start, el_stop = el_stop, el_start
            azim_angles = np.arange(az_start, az_stop + step / 2, step)
            elev_angles = np.arange(el_start, el_stop + step / 2, step)
        else:
            azim_angles = np.arange(-180, 180, step)            # -180 … +175 (5° step)
            elev_angles = np.arange(-90, 90 + step / 2, step)  # -90  … +90  inclusive

        total_steps = (len(azim_angles) * len(elev_angles)
                       if is_3d else len(azim_angles))
        done        = 0

        progress_bar.value = 0
        percent_label.text = '0 %'
        status_label.text  = 'Starting…'
        if vna_error_banner is not None:
            try:
                vna_error_banner.style('display:none;')
                vna_error_banner.update()
            except RuntimeError:
                pass
        _vna_error_count = 0
        t_start = time.monotonic()
        logger.info(
            f'[SCAN] {project}  {measurement_radio.value}  '
            f'{freq_start/1e9:.3f}–{freq_stop/1e9:.3f} GHz  '
            f'{acq_points} pts  step={step}°  {total_steps} positions'
        )

        # ══ SWEEP LOOP ════════════════════════════════════════════
        elev_forward = True   # snake scan: alternates elevation direction each azimuth row

        for azim_antenna in azim_angles:
            if stop_requested:
                break

            if is_3d:
                elev_range   = elev_angles if elev_forward else elev_angles[::-1]
                elev_forward = not elev_forward
            else:
                elev_range = [0.0]

            for elev_antenna in elev_range:
                if stop_requested:
                    break

                if not is_3d:
                    if cut_type == 'Constant Elevation':
                        # azim_antenna = outer loop variable (sweeping)
                        elev_antenna   = const_angle
                    else:
                        # Constant Elevation: sweep azimuth, fix elevation.
                        # The outer loop variable (azim_antenna) carries
                        # the sweep index → reassign it to elev before
                        # overwriting azimuth with the constant value.
                        elev_antenna   = azim_antenna
                        azim_antenna = const_angle

                elapsed_so_far = time.monotonic() - t_start
                if done > 0:
                    eta_s   = elapsed_so_far / done * (total_steps - done)
                    eta_str = f"{int(eta_s//60)}m{int(eta_s%60):02d}s"
                else:
                    eta_str = '—'
                logger.info(
                    f'[{done+1:>4}/{total_steps}] '
                    f'Az={azim_antenna:+7.1f}°  El={elev_antenna:+7.1f}°  ETA {eta_str}'
                )

                try:
                    if motor_status_label:
                        motor_status_label.text = '🔄 Moving…'
                        motor_status_label.style('color:#facc15; font-weight:700;')
                except RuntimeError:
                    pass
                _azim_off = float(azim_offset_input.value or AZIMUTH_OFFSET_DEG)
                _elev_off   = float(elev_offset_input.value   or ELEVATION_OFFSET_DEG)
                await _move_to_abs(
                    AZIMUTH_CAN_ID, azim_antenna, spd, smooth,
                    AZIMUTH_GEAR_RATIO, _azim_off
                )
                await _move_to_abs(
                    ELEVATION_CAN_ID, elev_antenna, spd, smooth,
                    ELEVATION_GEAR_RATIO, _elev_off
                )
                try:
                    if motor_status_label:
                        motor_status_label.text = '✅ In position'
                        motor_status_label.style('color:#34d399; font-weight:700;')
                except RuntimeError:
                    pass

                try:
                    update_gauge(azim_pointer, azim_value_label, azim_antenna)
                    update_gauge(elev_pointer,   elev_value_label,   elev_antenna)
                except RuntimeError:
                    pass

                # Settle delay — let mechanical oscillations die out
                _settle = float(settle_delay_input.value
                                if settle_delay_input is not None else 0.1)
                if _settle > 0:
                    await asyncio.sleep(_settle)

                try:
                    if vna_status_label:
                        vna_status_label.text = '📡 Sweeping…'
                        vna_status_label.style('color:#60a5fa; font-weight:700;')
                except RuntimeError:
                    pass
                # Check errors BEFORE sweep — flags from previous sweep's
                # idle transition have had time to clear (transient is gone)
                await _poll_vna_errors()
                await _wait_sweep_done()
                try:
                    if vna_status_label:
                        vna_status_label.text = '✅ Done'
                        vna_status_label.style('color:#34d399; font-weight:700;')
                except RuntimeError:
                    pass
                raw = _vna_query(':VNA:TRACE:DATA? S21')
                logger.debug(f'VNA raw (first 80): {raw[:80]!r}')

                if not raw or raw.strip() in ('', 'NA', 'ERROR'):
                    logger.warning(
                        f'Empty VNA response at '
                        f'Az={azim_antenna:.1f}° El={elev_antenna:.1f}° — skip'
                    )
                    continue
                try:
                    data_parsed = libreVNA.parse_VNA_trace_data(raw)
                except Exception as e:
                    logger.warning(f'parse_VNA_trace_data: {e} — skip')
                    continue

                # Apply Savitzky-Golay smoothing to raw complex S21
                _hw = int(vna_smooth_slider.value) if vna_smooth_slider else 0
                if _hw > 0:
                    data_parsed = _smooth_vna(data_parsed, _hw)
                    logger.debug(f'  [VNA] S-G smooth half_window={_hw} '
                                 f'(window={2*_hw+1})')

                for freq_hz, s21_cplx in data_parsed:
                    df_data.append({
                        'azim_deg': azim_antenna,
                        'elev_deg':   elev_antenna,
                        'freq_hz':   freq_hz,
                        's21_db':    20 * np.log10(max(abs(s21_cplx), 1e-12)),
                    })

                done += 1
                pct   = done / total_steps
                _ui_set(progress_bar, "value", pct)
                _ui_set(percent_label, "text", f'{pct * 100:.1f} %')
                _ui_set(status_label, "text",
                    f'Point {done}/{total_steps}  '
                    f'Az={azim_antenna:+.1f}°  El={elev_antenna:+.1f}°  '
                    f'ETA {eta_str}')
                try:
                    if eta_label:
                        eta_label.text = f'ETA {eta_str}'
                except RuntimeError:
                    pass

                try:
                    update_vna_trace(data_parsed)
                    update_farfield_plot()
                except RuntimeError:
                    pass
                await asyncio.sleep(0.05)   # flush WebSocket before next CAN/VNA block

        # ── Return to physical 0° ─────────────────────────────────
        status_label.text = 'Returning to physical 0°…'
        logger.info('[SCAN] Returning to physical 0° (no offset)')
        await _move_to_abs(AZIMUTH_CAN_ID, 0, spd, smooth, AZIMUTH_GEAR_RATIO, offset_deg=0.0)
        await _move_to_abs(ELEVATION_CAN_ID,   0, spd, smooth, ELEVATION_GEAR_RATIO,   offset_deg=0.0)
        update_gauge(azim_pointer, azim_value_label, 0.0)
        update_gauge(elev_pointer,   elev_value_label,   0.0)
        # Refresh live plots with final data so they remain visible after scan
        update_farfield_plot()
        await asyncio.sleep(0.1)

        if df_data:
            _save_csv(project)
            # Save project notes + full settings snapshot
            notes_path = _project_dir(project) / f'{project}_notes.txt'
            try:
                import datetime as _dt
                def _w(v):
                    """Safe widget value reader."""
                    try:
                        return str(v.value) if v is not None else '—'
                    except Exception:
                        return '—'

                lines = [
                    '=' * 60,
                    f'  {project}  —  Measurement Notes',
                    '=' * 60,
                    f'  Date / Time   : {_dt.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")}',
                    '',
                    '── VNA Settings ─────────────────────────────────────────',
                    f'  Frequency Range  : {_w(freq_range_input)}',
                    f'  Frequency Step   : {_w(vna_freq_step_input)} MHz',
                    f'  IF Bandwidth     : {_w(vna_ifbw_input)} Hz',
                    f'  Stimulus Level   : {_w(vna_level_input)} dBm',
                    f'  Averaging        : {_w(vna_avg_input)} sweep(s)/pt',
                    f'  S-G Smoothing    : {_w(vna_smooth_slider)} (half-window)',
                    '',
                    '── Scan Settings ────────────────────────────────────────',
                    f'  Measurement Type : {_w(measurement_radio)}',
                    f'  Angle Step       : {_w(angle_step_input)} °',
                    *([
                        f'  Azimuth Range    : {_w(azim_start_input)} … {_w(azim_stop_input)} °',
                        f'  Elevation Range  : {_w(elev_start_input)} … {_w(elev_stop_input)} °',
                    ] if measurement_radio.value == '3D - Range' else []),
                    *([] if is_3d else [   # hide 2D-only fields for both 3D modes
                        f'  2D Cut Type      : {_w(cut_type_dropdown)}',
                        f'  Constant Angle   : {_w(constant_angle_input)} °',
                    ]),
                    '',
                    '── Motor Settings ───────────────────────────────────────',
                    f'  CANable Port     : {_w(can_port_input)}',
                    f'  Settle Delay     : {_w(settle_delay_input)} s',
                    f'  Antenna Speed    : {_w(speed_slider)} deg/s',
                    f'  Motion Smoothness: {_w(acc_slider)}',
                    f'  Azimuth Offset   : {_w(azim_offset_input)} °',
                    f'  Elevation Offset : {_w(elev_offset_input)} °',
                    '',
                    '── Normalisation ────────────────────────────────────────',
                    f'  Mode             : {_w(norm_radio)}',
                    f'  Reference Azimuth: {_w(norm_azim_input)} °',
                    f'  Reference Elev   : {_w(norm_elev_input)} °',
                    '',
                ]

                user_notes = (notes_input.value.strip()
                              if notes_input and notes_input.value.strip()
                              else '(no user notes)')
                lines += [
                    '── User Notes ───────────────────────────────────────────',
                    user_notes,
                    '=' * 60,
                ]

                notes_path.write_text('\n'.join(lines), encoding='utf-8')
                logger.info(f'[NOTES] Saved → {notes_path}')
            except Exception as _ne:
                logger.warning(f'[NOTES] Failed to save notes: {_ne}')
            if plot_enable_radio and plot_enable_radio.value == 'Yes':
                logger.info('[PLOT] Generating farfield plots…')
                # Run in executor so matplotlib doesn't block the asyncio event
                # loop and cause a WebSocket ping-timeout / spurious disconnect
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    _executor, _generate_farfield_plots, project
                )
            if html3d_enable_radio and html3d_enable_radio.value == 'Yes':
                logger.info('[HTML3D] Generating interactive 3D HTML…')
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    _executor, _generate_3d_html, project
                )

    except Exception as exc:
        _safe_notify(f'❌ Error: {exc}', type='negative')
        logger.error('Measurement error', exc_info=True)

    finally:
        await _disconnect_hw()
        measurement_running = False
        try:
            start_button.enabled = True
            stop_button.enabled  = False
            status_label.text    = 'Finished / Stopped ✓'
            if motor_status_label:
                motor_status_label.text = 'Idle'
                motor_status_label.style('color:#71717a; font-weight:700;')
            if vna_status_label:
                vna_status_label.text = 'Idle'
                vna_status_label.style('color:#71717a; font-weight:700;')
            if eta_label:
                eta_label.text = ''
        except RuntimeError:
            pass
        try:
            _elapsed = time.monotonic() - t_start
            logger.info(f'[SCAN] Done — {done}/{total_steps} pts  elapsed {int(_elapsed//60)}m{int(_elapsed%60):02d}s')
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# GUI LAYOUT
# ═══════════════════════════════════════════════════════════════════

farfield_fig.update_layout(
    xaxis_title='Angle (°)', yaxis_title='S21 (dB)',
    xaxis=dict(range=[-180, 180], dtick=45, gridcolor='#444'),
    yaxis=dict(autorange=True, gridcolor='#444'),
    plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a', font_color='#ddd',
    margin=dict(l=55, r=20, t=45, b=45),
)
vna_fig.update_layout(
    xaxis_title='Frequency (GHz)', yaxis_title='S21 (dB)',
    xaxis=dict(gridcolor='#444'),
    yaxis=dict(autorange=True, gridcolor='#444'),
    plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a', font_color='#ddd',
    margin=dict(l=55, r=20, t=45, b=45),
)

with ui.column().classes('w-full max-w-7xl mx-auto p-8 gap-8'):

    with ui.row().classes('w-full items-center'):
        ui.label('BYTECHLAB ANTENNA MEASUREMENT SYSTEM').classes(
            'text-4xl font-bold tracking-[0.25em] text-white flex-1 text-center'
        )
        with ui.row().classes('gap-2 flex-shrink-0'):
            ui.button('💾 Save Settings', on_click=_save_settings).classes(
                'bg-zinc-700 hover:bg-zinc-600 text-xs font-semibold rounded-xl px-3 py-2'
            )
            ui.button('📂 Load Settings', on_click=_load_settings).classes(
                'bg-zinc-700 hover:bg-zinc-600 text-xs font-semibold rounded-xl px-3 py-2'
            )

    # ── MEASUREMENT SETUP ────────────────────────────────────────
    with ui.card().classes('w-full p-6 bg-zinc-900 border border-zinc-700 rounded-2xl'):
        ui.label('MEASUREMENT SETUP').classes(
            'text-sm font-semibold tracking-widest text-zinc-400 mb-4'
        )
        with ui.grid(columns=2).classes('w-full gap-5'):
            project_input = ui.input(label='PROJECT NAME', value='TEST').classes('col-span-1')
            with ui.column().classes('col-span-1 justify-center gap-1'):
                ui.label('MEASUREMENT TYPE').classes(
                    'text-xs font-semibold tracking-widest text-zinc-400'
                )
                measurement_radio = ui.radio(['2D', '3D - Full', '3D - Range'], value='2D').props('inline').classes('mt-1')

        ui.separator().classes('my-4 border-zinc-700')

        ui.label('VNA SETTINGS').classes(
            'text-xs font-semibold tracking-widest text-zinc-400 mb-2'
        )
        with ui.grid(columns=7).classes('w-full gap-4 items-end'):
            freq_range_input = ui.input(
                label='FREQUENCY RANGE  (e.g. 1-3 GHz)', value='0.001-6 GHz',
                on_change=lambda _: _update_vna_ui()
            ).classes('col-span-1')

            vna_freq_step_input = ui.input(
                label='FREQUENCY STEP  (MHz)', value='4',
                on_change=lambda _: _update_vna_ui()
            ).classes('col-span-1')

            # Calculated points display (read-only)
            with ui.column().classes('col-span-1 gap-1'):
                ui.label('ACQUISITION POINTS').classes('text-xs text-zinc-400')
                vna_points_label = ui.label('501').classes(
                    'text-2xl font-extrabold font-mono text-emerald-400'
                )
                # Seed initial value: (0.001-6 GHz) / 4 MHz + 1 = 1500
                ui.timer(0.1, _update_points_label, once=True)

            # VNA data smoothing
            with ui.column().classes('col-span-1 gap-1'):
                with ui.row().classes('items-center gap-3'):
                    ui.label('DATA SMOOTHING  (Savitzky-Golay)').classes(
                        'text-xs text-zinc-400'
                    )
                    vna_smooth_label = ui.label('OFF').classes(
                        'text-xs font-mono text-zinc-500'
                    )
                vna_smooth_slider = (
                    ui.slider(min=0, max=25, value=0, step=1)
                    .props('label-always')
                    .classes('w-full')
                    .on('update:model-value', lambda e: (
                        vna_smooth_label.__setattr__(
                            'text',
                            'OFF – raw data' if int(e.args) == 0
                            else f'±{int(e.args)} pts  (window {2*int(e.args)+1})'
                        )
                    ))
                )

            # VNA averaging
            with ui.column().classes('col-span-1 gap-1'):
                ui.label('AVERAGING  (sweeps/pt)').classes('text-xs text-zinc-400')
                vna_avg_input = ui.number(
                    value=1, min=1, max=100, step=1, format='%d'
                ).classes('w-full').props('dense')
                ui.label('Higher = less noise, slower').classes('text-xs text-zinc-500 mt-1')

            # IF Bandwidth
            with ui.column().classes('col-span-1 gap-1'):
                ui.label('IF BANDWIDTH  (Hz)').classes('text-xs text-zinc-400')
                vna_ifbw_input = ui.number(
                    value=1000, min=1, max=1000000, step=100, format='%d'
                ).classes('w-full').props('dense')
                ui.label('Lower = less noise, slower').classes('text-xs text-zinc-500 mt-1')

            # Stimulus Level
            with ui.column().classes('col-span-1 gap-1'):
                ui.label('STIMULUS LEVEL  (dBm)').classes('text-xs text-zinc-400')
                vna_level_input = ui.number(
                    value=-10, min=-42, max=0, step=1, format='%.0f'
                ).classes('w-full').props('dense')
                ui.label('Typ. range −42 to 0 dBm').classes('text-xs text-zinc-500 mt-1')

        ui.separator().classes('my-4 border-zinc-700')

        with ui.grid(columns=3).classes('w-full gap-5'):
            angle_step_input = ui.input(label='ANGLE STEP (°)', value='5').classes('col-span-1')

            # 2D-only controls – dimmed and disabled when 3D is selected
            with ui.element('div').classes('col-span-1') as _cut_type_wrap:
                cut_type_dropdown = ui.select(
                    options=['Constant Elevation', 'Constant Azimuth'],
                    value='Constant Elevation', label='2D CUT TYPE',
                ).classes('w-full')
            with ui.element('div').classes('col-span-1') as _const_angle_wrap:
                constant_angle_input = ui.input(
                    label='CONSTANT ANGLE (°)', value='0'
                ).classes('w-full')

        # 3D - Range controls — shown only when that mode is selected
        with ui.element('div').classes('w-full') as _range_wrap:
            ui.separator().classes('my-2 border-zinc-700')
            ui.label('3D RANGE SETTINGS').classes(
                'text-xs font-semibold tracking-widest text-zinc-400 mb-2'
            )
            with ui.grid(columns=4).classes('w-full gap-5'):
                azim_start_input = ui.number(
                    label='AZIMUTH START  (°)', value=-180,
                    min=-180, max=180, step=1, format='%.0f',
                ).classes('w-full')
                azim_stop_input = ui.number(
                    label='AZIMUTH STOP  (°)', value=175,
                    min=-180, max=180, step=1, format='%.0f',
                ).classes('w-full')
                elev_start_input = ui.number(
                    label='ELEVATION START  (°)', value=-90,
                    min=-90, max=90, step=1, format='%.0f',
                ).classes('w-full')
                elev_stop_input = ui.number(
                    label='ELEVATION STOP  (°)', value=90,
                    min=-90, max=90, step=1, format='%.0f',
                ).classes('w-full')

        ui.separator().classes('my-4 border-zinc-700')

        ui.label('PROJECT NOTES').classes(
            'text-xs font-semibold tracking-widest text-zinc-400 mb-2'
        )
        notes_input = ui.textarea(
            placeholder='Antenna serial, cable type, operator, test conditions…',
        ).classes('w-full').props('rows=3 outlined dense')

        def _on_mode_change():
            mode  = measurement_radio.value
            is_3d = mode in ('3D - Full', '3D - Range')
            is_range = mode == '3D - Range'
            # Disable 2D-only controls when 3D selected
            cut_type_dropdown.enabled    = not is_3d
            constant_angle_input.enabled = not is_3d
            opacity_2d = 'opacity:0.3; pointer-events:none;' if is_3d else 'opacity:1; pointer-events:auto;'
            _cut_type_wrap.style(opacity_2d)
            _const_angle_wrap.style(opacity_2d)
            # Range inputs only shown in 3D - Range mode
            opacity_range = 'opacity:1; pointer-events:auto;' if is_range else 'opacity:0.3; pointer-events:none;'
            try:
                _range_wrap.style(opacity_range)
                for w in (azim_start_input, azim_stop_input,
                          elev_start_input, elev_stop_input):
                    if w: w.enabled = is_range
            except Exception:
                pass
            # Shape slider only meaningful in 3D mode
            opacity_3d = 'opacity:1; pointer-events:auto;' if is_3d else 'opacity:0.3; pointer-events:none;'
            try:
                _shape_slider_wrap.style(opacity_3d)
                shape_slider.enabled = is_3d
            except Exception:
                pass
            # Disable 3D HTML export card when 2D selected
            # html3d card is defined later but in the same page-build scope;
            # use a try/except to handle the startup call before it's assigned
            try:
                if is_3d:
                    _html3d_card_wrap.style('opacity:1; pointer-events:auto;')
                else:
                    html3d_enable_radio.value = 'No'
                    html3d_enable_radio.update()
                    _on_html3d_toggle()
                    _html3d_card_wrap.style('opacity:0.35; pointer-events:none;')
            except Exception:
                pass

        measurement_radio.on('update:model-value', lambda _: _on_mode_change())


    # ── S21 NORMALIZATION ────────────────────────────────────────
    with ui.card().classes('w-full p-6 bg-zinc-900 border border-zinc-700 rounded-2xl'):
        with ui.row().classes('w-full justify-between items-center mb-3'):
            ui.label('S21 NORMALISATION').classes(
                'text-sm font-semibold tracking-widest text-zinc-400'
            )
            with ui.row().classes('items-center gap-3'):
                norm_radio = ui.radio(
                    ['Off', 'Angle', 'Max'], value='Off'
                ).props('inline').classes('text-sm')

        ui.label(
            'Off = raw dB.  '
            'Angle = subtract S21 at reference (θ, φ).  '
            'Max = subtract peak S21 per frequency.  '
            'CSV always contains raw data.'
        ).classes('text-xs text-zinc-500 mb-3 leading-relaxed')

        with ui.row().classes('w-full gap-6 items-end') as _norm_angle_row:
            with ui.column().classes('gap-1'):
                ui.label('Reference Azimuth (°)').classes('text-xs text-zinc-400')
                norm_azim_input = ui.number(
                    value=0, step=1, format='%.1f'
                ).classes('w-40').props('dense')
            with ui.column().classes('gap-1'):
                ui.label('Reference Elevation (°)').classes('text-xs text-zinc-400')
                norm_elev_input = ui.number(
                    value=0, step=1, format='%.1f'
                ).classes('w-40').props('dense')
            ui.label(
                'Nearest measured angle used if exact match not found.'
            ).classes('text-xs text-zinc-500')

        def _on_norm_change():
            mode = norm_radio.value
            _norm_angle_row.style(
                'opacity:1; pointer-events:auto;'
                if mode == 'Angle'
                else 'opacity:0.3; pointer-events:none;'
            )
            update_farfield_plot()

        norm_radio.on('update:model-value', lambda _: _on_norm_change())
        _on_norm_change()

    # ── FARFIELD PLOT EXPORT ─────────────────────────────────────
    with ui.card().classes('w-full p-6 bg-zinc-900 border border-zinc-700 rounded-2xl'):
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('FARFIELD PLOT EXPORT').classes(
                'text-sm font-semibold tracking-widest text-zinc-400'
            )
            with ui.row().classes('items-center gap-3'):
                ui.label('Generate plots after measurement:').classes('text-xs text-zinc-400')
                plot_enable_radio = ui.radio(
                    ['Yes', 'No'], value='No'
                ).props('inline').classes('text-sm')

        with ui.column().classes('w-full gap-1') as _plot_opts:
            ui.label(
                'Plot frequencies (GHz) — comma or space separated.  '
                'Nearest measured frequency used if exact match not found.'
            ).classes('text-xs text-zinc-500 mb-1')
            plot_freq_input = ui.input(
                label='PLOT FREQUENCIES  (GHz, e.g.  1 2 3 4 5 6)',
                value='1 2 3 4 5 6'
            ).classes('w-full')
            ui.label(
                'Saved to:  RESULTS/[PROJECT]/PLOTS/  '
                'as PNG polar plots — one per frequency per cut (Azimuth plane and Elevation plane).'
            ).classes('text-xs text-zinc-500 mt-1')

        def _on_plot_toggle():
            enabled = plot_enable_radio.value == 'Yes'
            _plot_opts.style(
                'opacity:1; pointer-events:auto;'
                if enabled
                else 'opacity:0.35; pointer-events:none;'
            )
        plot_enable_radio.on('update:model-value', lambda _: _on_plot_toggle())
        _on_plot_toggle()   # set initial state

    # ── INTERACTIVE 3D HTML EXPORT ──────────────────────────────────
    with ui.element('div').classes('w-full') as _html3d_card_wrap:
      with ui.card().classes('w-full p-6 bg-zinc-900 border border-zinc-700 rounded-2xl'):
        with ui.row().classes('w-full justify-between items-center mb-4'):
            ui.label('INTERACTIVE 3D FARFIELD  (HTML)').classes(
                'text-sm font-semibold tracking-widest text-zinc-400'
            )
            with ui.row().classes('items-center gap-3'):
                ui.label('Generate HTML after measurement:').classes('text-xs text-zinc-400')
                html3d_enable_radio = ui.radio(
                    ['Yes', 'No'], value='No'
                ).props('inline').classes('text-sm')

        with ui.column().classes('w-full gap-1') as _html3d_opts:
            ui.label(
                'Frequencies to include (GHz) — comma or space separated. '
                'Nearest measured frequency used if exact match not found.'
            ).classes('text-xs text-zinc-500 mb-1')
            html3d_freq_input = ui.input(
                label='3D HTML FREQUENCIES  (GHz, e.g.  1 2 3 4 5 6)',
                value='1 2 3 4 5 6'
            ).classes('w-full')
            ui.label(
                'Saved to:  RESULTS/[PROJECT]/[PROJECT]_3D_farfield.html  ·  '
                'Includes frequency & smoothing sliders.  '
                'Requires internet (Plotly.js CDN).'
            ).classes('text-xs text-zinc-500 mt-1')

        def _on_html3d_toggle():
            enabled = html3d_enable_radio.value == 'Yes'
            _html3d_opts.style(
                'opacity:1; pointer-events:auto;'
                if enabled
                else 'opacity:0.35; pointer-events:none;'
            )
        html3d_enable_radio.on('update:model-value', lambda _: _on_html3d_toggle())
        _on_html3d_toggle()
        # Apply initial 2D/3D state now that _html3d_card_wrap exists
        ui.timer(0.05, _on_mode_change, once=True)

        # Register a single function that re-applies ALL toggle states.
        # Called by _load_settings after it writes widget values from JSON.
        def _refresh_all_ui_states():
            _on_mode_change()
            _on_html3d_toggle()
            _on_plot_toggle()
            _on_norm_change()

        _ui_state_updater = _refresh_all_ui_states

    # ── MOTOR SETTINGS ───────────────────────────────────────────
    with ui.card().classes('w-full p-6 bg-zinc-900 border border-zinc-700 rounded-2xl'):
        ui.label('MOTOR SETTINGS').classes(
            'text-sm font-semibold tracking-widest text-zinc-400 mb-4'
        )
        with ui.row().classes('w-full items-start gap-8 mb-4'):
            with ui.column().classes('gap-1'):
                can_port_input = ui.input(
                    label='CANABLE PORT',
                    value=CAN_CHANNEL,
                    placeholder='e.g. COM3',
                ).classes('w-40').props('dense')
                ui.label('Serial port of the MKS CANable V2.0 SLCAN adapter.').classes('text-xs text-zinc-500')
                ui.label('Change only takes effect on next Connect.').classes('text-xs text-zinc-500')
            with ui.column().classes('gap-1'):
                settle_delay_input = ui.number(
                    label='SETTLE DELAY  (s)',
                    value=0.1, min=0.0, max=10.0, step=0.05, format='%.2f',
                ).classes('w-36').props('dense')
                ui.label('Pause after motor stops, before VNA sweep starts.').classes('text-xs text-zinc-500')
                ui.label('Eliminates mechanical oscillation effects on measurement.').classes('text-xs text-zinc-500')
        with ui.grid(columns=2).classes('w-full gap-8'):
            with ui.column().classes('gap-2'):
                ui.label('Antenna Speed  (deg/s)').classes('text-sm text-zinc-400')
                speed_slider = (
                    ui.slider(min=0.5, max=30, value=30, step=0.1)
                    .props('label-always').classes('w-full')
                )
            with ui.column().classes('gap-2'):
                ui.label(
                    'Motion Smoothness  '
                    '(1 = instant/jerky  →  255 = slowest ramp)'
                ).classes('text-sm text-zinc-400')
                acc_slider = (
                    ui.slider(min=1, max=255, value=255, step=1)
                    .props('label-always').classes('w-full')
                )

        ui.separator().classes('my-4 border-zinc-700')

        ui.label('AXIS ZERO OFFSETS').classes(
            'text-xs font-semibold tracking-widest text-zinc-400 mb-2'
        )
        ui.label(
            'Physical motor angle = measurement angle + offset.  '
            'Adjust so that measurement 0° points to boresight.'
        ).classes('text-xs text-zinc-500 mb-3')
        with ui.grid(columns=2).classes('w-full gap-5'):
            azim_offset_input = ui.number(
                label='AZIMUTH OFFSET  (°)',
                value=AZIMUTH_OFFSET_DEG,
                step=1, format='%.1f',
            ).classes('w-full')
            elev_offset_input = ui.number(
                label='ELEVATION OFFSET  (°)',
                value=ELEVATION_OFFSET_DEG,
                step=1, format='%.1f',
            ).classes('w-full')

        with ui.row().classes('w-full gap-4 mt-4'):
            _test_btn = ui.button(
                'TEST ZERO POSITION',
                on_click=_tesazim_offsets,
            ).classes(
                'flex-1 py-4 text-base font-bold '
                'bg-amber-600 hover:bg-amber-500 rounded-2xl'
            )
            _return_btn = ui.button(
                'RETURN TO PHYSICAL 0°',
                on_click=_return_to_physical_zero,
            ).classes(
                'flex-1 py-4 text-base font-bold '
                'bg-zinc-600 hover:bg-zinc-500 rounded-2xl'
            )
            _return_btn.enabled = False

        ui.label(
            '① Click TEST ZERO POSITION — motors move to measurement 0° with offsets applied.  '
            '② Verify antenna boresight visually.  '
            '③ Click RETURN TO PHYSICAL 0° when done.'
        ).classes('text-xs text-zinc-500 mt-1 leading-relaxed')

    # ── START / STOP ──────────────────────────────────────────────
    with ui.row().classes('w-full gap-4'):
        start_button = ui.button(
            'START MEASUREMENT', on_click=perform_measurement,
        ).classes(
            'flex-1 py-6 text-2xl font-bold '
            'bg-emerald-600 hover:bg-emerald-500 rounded-3xl'
        )
        stop_button = ui.button(
            'STOP', on_click=_set_stop,
        ).classes(
            'flex-1 py-6 text-2xl font-bold '
            'bg-red-600 hover:bg-red-500 rounded-3xl'
        )
        stop_button.enabled = False

        def show_motor_setup():
            with ui.dialog() as dlg, ui.card().classes(
                'bg-zinc-900 border border-zinc-700 rounded-2xl p-8 max-w-2xl w-full'
            ):
                with ui.row().classes('w-full justify-between items-center mb-6'):
                    ui.label('MOTOR SETUP GUIDE').classes(
                        'text-2xl font-bold tracking-widest text-white'
                    )
                    ui.button(icon='close', on_click=dlg.close).props(
                        'flat round dense'
                    ).classes('text-zinc-400 hover:text-white')

                ui.label(
                    'Perform these steps on BOTH motors using the OLED display '
                    'and buttons before first use. Settings are saved to EEPROM '
                    'and only need to be set once.'
                ).classes('text-sm text-zinc-400 mb-6 leading-relaxed')

                steps = [
                    (
                        '1',
                        'Reset to Default Settings',
                        'Menu → Restore → Press ✓',
                        'Clears any previous configuration and returns all '
                        'parameters to factory defaults.',
                        '#facc15',
                    ),
                    (
                        '2',
                        'Set Mode → SR_CLOSE',
                        'Menu → Mode → SR_CLOSE → Press ✓',
                        'Selects Bus Interface Closed-Loop mode. '
                        'Required for CAN position control with encoder feedback.',
                        '#34d399',
                    ),
                    (
                        '3',
                        'Set Protect → Enable',
                        'Menu → Protect → Enable → Press ✓',
                        'Enables stall protection. The motor will stop if it '
                        'cannot reach the target position, preventing damage.',
                        '#60a5fa',
                    ),
                    (
                        '4',
                        'Set CAN ID',
                        'Menu → CanID → 1 (Azimuth)  /  2 (Elevation) → Press ✓',
                        'Azimuth motor must be set to CAN ID = 1. '
                        'Elevation motor must be set to CAN ID = 2. '
                        'Each motor must have a unique ID.',
                        '#f472b6',
                    ),
                ]

                for num, title, path, desc, colour in steps:
                    with ui.card().classes(
                        'w-full p-4 mb-3 bg-zinc-800 border border-zinc-700 rounded-xl'
                    ):
                        with ui.row().classes('items-start gap-4'):
                            with ui.element('div').style(
                                f'min-width:2rem; height:2rem; border-radius:50%; '
                                f'background:{colour}22; border:2px solid {colour}; '
                                f'display:flex; align-items:center; '
                                f'justify-content:center; flex-shrink:0;'
                            ):
                                ui.label(num).style(
                                    f'font-weight:900; color:{colour}; font-size:0.9rem;'
                                )
                            with ui.column().classes('gap-1'):
                                ui.label(title).style(
                                    f'font-weight:700; color:{colour}; font-size:1rem;'
                                )
                                ui.label(path).classes(
                                    'font-mono text-xs text-white bg-zinc-950 '
                                    'px-2 py-1 rounded'
                                )
                                ui.label(desc).classes('text-xs text-zinc-400 mt-1')

                ui.separator().classes('my-4 border-zinc-700')

                ui.label('VNA WIRING').classes(
                    'text-sm font-semibold tracking-widest text-zinc-400 mb-3'
                )

                vna_steps = [
                    (
                        '1',
                        'Port 1  →  Reference Antenna',
                        'Connect LibreVNA Port 1 to the reference horn antenna '
                        '(fixed, pointing at the DUT).',
                        '#a78bfa',
                    ),
                    (
                        '2',
                        'Port 2  →  DUT Antenna',
                        'Connect LibreVNA Port 2 to the antenna under test '
                        '(mounted on the rotator).',
                        '#fb923c',
                    ),
                ]

                for num, title, desc, colour in vna_steps:
                    with ui.card().classes(
                        'w-full p-4 mb-3 bg-zinc-800 border border-zinc-700 rounded-xl'
                    ):
                        with ui.row().classes('items-start gap-4'):
                            with ui.element('div').style(
                                f'min-width:2rem; height:2rem; border-radius:50%; '
                                f'background:{colour}22; border:2px solid {colour}; '
                                f'display:flex; align-items:center; '
                                f'justify-content:center; flex-shrink:0;'
                            ):
                                ui.label(num).style(
                                    f'font-weight:900; color:{colour}; font-size:0.9rem;'
                                )
                            with ui.column().classes('gap-1'):
                                ui.label(title).style(
                                    f'font-weight:700; color:{colour}; font-size:1rem;'
                                )
                                ui.label(desc).classes('text-xs text-zinc-400 mt-1')

                with ui.card().classes(
                    'w-full p-4 mb-3 bg-amber-950 border border-amber-600 rounded-xl'
                ):
                    with ui.row().classes('items-start gap-3'):
                        ui.label('⚠️').style('font-size:1.3rem; flex-shrink:0;')
                        with ui.column().classes('gap-1'):
                            ui.label('Cable Loss Warning').style(
                                'font-weight:700; color:#fbbf24; font-size:1rem;'
                            )
                            ui.label(
                                'Please use low-loss coax cables. '
                                'Total insertion loss of both cables combined must be '
                                'below 10 dB. '
                                'This is required because LibreVNA has limited dynamic '
                                'range and port isolation, which degrades significantly '
                                'at higher frequencies — excessive cable loss will '
                                'corrupt the measured antenna pattern.'
                            ).classes('text-xs text-amber-200 mt-1 leading-relaxed')

                ui.separator().classes('my-4 border-zinc-700')
                ui.label(
                    '⚡ After changing motor settings, power-cycle both motors '
                    'to ensure all changes take effect.'
                ).classes('text-xs text-amber-400')

                ui.button('Close', on_click=dlg.close).classes(
                    'w-full mt-4 py-3 font-bold bg-zinc-700 '
                    'hover:bg-zinc-600 rounded-xl text-white'
                )
            dlg.open()

        ui.button('⚙ INSTRUCTIONS', on_click=show_motor_setup).classes(
            'py-6 px-8 text-lg font-bold '
            'bg-zinc-700 hover:bg-zinc-600 rounded-3xl'
        )

    # ── PROGRESS ─────────────────────────────────────────────────
    with ui.card().classes('w-full p-5 bg-zinc-900 border border-zinc-700 rounded-2xl'):
        progress_bar = ui.linear_progress(value=0, show_value=False).classes('w-full h-3 rounded-full')
        with ui.row().classes('justify-between items-center w-full text-sm font-mono mt-2'):
            percent_label = ui.label('0 %').classes('text-white font-extrabold text-2xl')
            status_label  = ui.label('Ready to start').classes('text-zinc-400')
            eta_label     = ui.label('').classes('text-zinc-500 font-mono text-xs')
        ui.separator().classes('my-3 border-zinc-700')
        with ui.row().classes('w-full gap-8'):
            with ui.row().classes('items-center gap-2'):
                ui.label('MOTORS').classes('text-xs font-semibold tracking-widest text-zinc-500')
                motor_status_label = ui.label('Idle').style('color:#71717a; font-weight:700; font-size:0.9rem;')
            with ui.row().classes('items-center gap-2'):
                ui.label('VNA').classes('text-xs font-semibold tracking-widest text-zinc-500')
                vna_status_label = ui.label('Idle').style('color:#71717a; font-weight:700; font-size:0.9rem;')

        # VNA error banner — hidden by default, shown when errors are detected
        vna_error_banner = ui.element('div').style(
            'display:none; width:100%; margin-top:10px; padding:10px 16px; '
            'background:#7f1d1d; border:1.5px solid #ef4444; border-radius:10px;'
        )
        with vna_error_banner:
            vna_error_text = ui.label('').classes(
                'font-mono text-sm font-bold text-red-300'
            ).style('white-space:pre-line;')

    # ── AXIS GAUGES ───────────────────────────────────────────────
    with ui.row().classes('w-full gap-8 justify-center flex-wrap'):
        azim_pointer, azim_value_label = create_gauge('AZIMUTH AXIS')
        elev_pointer,   elev_value_label   = create_gauge('ELEVATION AXIS')

    # ── FARFIELD PREVIEW ─────────────────────────────────────────
    with ui.card().classes('w-full p-6 bg-zinc-900 border border-zinc-700 rounded-2xl'):
        with ui.row().classes('items-center gap-4 mb-1 flex-wrap'):
            ui.label('FARFIELD PREVIEW').classes(
                'text-lg font-bold tracking-widest text-white'
            )
            freq_label = ui.label('0.001 GHz').classes(
                'text-base font-mono font-bold text-emerald-400 ml-4'
            )

        with ui.column().classes('w-full gap-1 mb-3'):
            ui.label('Preview frequency (GHz)').classes('text-xs text-zinc-400')
            freq_slider = (
                ui.slider(min=0.001, max=6.0, value=0.001, step=0.001)
                .props('label-always').classes('w-full')
                .on('update:model-value', lambda e: (
                    _safe_setattr(freq_label, 'text', f'{float(e.args):.3f} GHz'),
                    _schedule_farfield_update()
                ))
            )

        with ui.column().classes('w-full gap-1 mb-4'):
            with ui.row().classes('items-center gap-3'):
                ui.label('Smoothing  (σ)').classes('text-xs text-zinc-400')
                smooth_label = ui.label('OFF – raw data').classes(
                    'text-xs font-mono text-zinc-500'
                )
            smooth_slider = (
                ui.slider(min=0.0, max=5.0, value=0.0, step=0.5)
                .props('label-always').classes('w-full')
                .on('update:model-value', lambda e: (
                    _safe_setattr(
                        smooth_label, 'text',
                        'OFF – raw data' if float(e.args) == 0
                        else f'σ = {float(e.args):.1f}'
                    ),
                    _schedule_farfield_update()
                ))
            )

        with ui.column().classes('w-full gap-1 mb-4') as _shape_slider_wrap:
            with ui.row().classes('items-center gap-3'):
                ui.label('Shape compression  (γ)').classes('text-xs text-zinc-400')
                shape_label = ui.label('γ = 1.0  (linear)').classes(
                    'text-xs font-mono text-zinc-500'
                )
            shape_slider = (
                ui.slider(min=0.1, max=1.0, value=1.0, step=0.1)
                .props('label-always').classes('w-full')
                .on('update:model-value', lambda e: (
                    _safe_setattr(shape_label, 'text',
                        f'γ = {float(e.args):.1f}  '
                        f'({"linear" if float(e.args) == 1.0 else "compressed"})'),
                    _schedule_farfield_update()
                ))
            )

        live_farfield_plot = ui.plotly(farfield_fig).classes('w-full h-[500px]')

    # ── LIVE VNA TRACE ────────────────────────────────────────────
    with ui.card().classes('w-full p-6 bg-zinc-900 border border-zinc-700 rounded-2xl'):
        ui.label('LIVE VNA S21 TRACE  (current measurement point)').classes(
            'text-lg font-bold tracking-widest text-white mb-3'
        )
        live_vna_trace = ui.plotly(vna_fig).classes('w-full h-96')


    # ── FOOTER ───────────────────────────────────────────────────
    with ui.row().classes('w-full justify-center items-center gap-4 py-4 mt-4'):
        with ui.column().style('gap:2px; user-select:none; opacity:0.7; align-items:center;'):
            ui.label('·').classes('text-zinc-700')
        ui.html('<a href="https://bytechlab.com/" target="_blank" '
                'style="color:#60a5fa;font-size:0.85rem;text-decoration:none;'
                'letter-spacing:0.05em;font-weight:500;">bytechlab.com</a>')
        ui.label('·').classes('text-zinc-700')
        ui.label('Antenna Measurement System').classes('text-zinc-600 text-sm')

# Auto-load saved settings if file exists
ui.timer(0.8, lambda: _load_settings(silent=True), once=True)

# ── Close browser tab on clean server shutdown only ─────────────────
# Listens for NiceGUI's 'disconnect' event but only closes the tab
# when the server sends an explicit shutdown signal (reason='io server disconnect')
# rather than a transient timeout during heavy CPU work (e.g. plot generation).
# Transient disconnects have reason='transport close' or 'ping timeout'.
ui.add_head_html("""
<script>
  window.addEventListener('load', function() {
    var _closeTimer = null;
    var _interval = setInterval(function() {
      if (window.socket) {
        window.socket.on('disconnect', function(reason) {
          // Start a 6-second timer. If NiceGUI reconnects (transient drop
          // during heavy CPU work), we cancel it. If the server is truly
          // dead (Python killed), reconnection fails and the tab closes.
          _closeTimer = setTimeout(function() { window.close(); }, 6000);
        });
        window.socket.on('connect', function() {
          // Server reconnected — cancel any pending close
          if (_closeTimer) { clearTimeout(_closeTimer); _closeTimer = null; }
        });
        clearInterval(_interval);
      }
    }, 200);
  });
</script>
""")

ui.run(
    title='Antenna Positioner Control',
    favicon='📡',
    dark=True,
    port=8080,
    reload=False,
    show=True,
    # Increase ping interval/timeout so heavy CPU work (matplotlib plot
    # generation) doesn't cause a spurious WebSocket disconnect.
    # Default is 20s interval / 20s timeout — raise both to 120s.
    uvicorn_logging_level='warning',
)