@echo off
:: ============================================================
::  BYTECHLAB Antenna Rotator Controller
::  Python package installer  –  Windows
::  Run this script once after a fresh Python installation.
::  Requires Python 3.10+ and pip available on PATH.
:: ============================================================

echo.
echo  =====================================================
echo   BYTECHLAB Antenna Rotator  --  Package Installer
echo  =====================================================
echo.

:: ── Check Python is available ────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found on PATH.
    echo          Install Python 3.10+ from https://www.python.org/downloads/
    echo          and make sure to check "Add Python to PATH" during setup.
    echo.
    pause
    exit /b 1
)

echo  [INFO] Python found:
python --version
echo.

:: ── Upgrade pip first ────────────────────────────────────────
echo  [1/8] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
echo.

:: ── NiceGUI  (UI framework) ──────────────────────────────────
echo  [2/8] Installing nicegui...
pip install nicegui
if errorlevel 1 goto :error
echo.

:: ── python-can + pyserial  (CAN bus / SLCAN) ─────────────────
echo  [3/8] Installing python-can...
pip install python-can
if errorlevel 1 goto :error
echo  [4/8] Installing pyserial  (required by python-can SLCAN backend)...
pip install pyserial
if errorlevel 1 goto :error
echo.

:: ── NumPy / Pandas ───────────────────────────────────────────
echo  [5/8] Installing numpy and pandas...
pip install numpy pandas
if errorlevel 1 goto :error
echo.

:: ── SciPy  (Savitzky-Golay + Gaussian smoothing) ─────────────
echo  [6/8] Installing scipy...
pip install scipy
if errorlevel 1 goto :error
echo.

:: ── Matplotlib  (PNG polar / 1D plots) ───────────────────────
echo  [7/8] Installing matplotlib...
pip install matplotlib
if errorlevel 1 goto :error
echo.

:: ── Plotly  (interactive 3D farfield HTML) ───────────────────
echo  [8/8] Installing plotly...
pip install plotly
if errorlevel 1 goto :error
echo.

:: ── Summary ──────────────────────────────────────────────────
echo  =====================================================
echo   All packages installed successfully.
echo.
echo   Installed:
echo     nicegui        -- web UI framework
echo     python-can     -- CAN bus communication
echo     pyserial       -- SLCAN serial port backend
echo     numpy          -- numerical arrays
echo     pandas         -- data handling / CSV export
echo     scipy          -- signal smoothing (Savitzky-Golay)
echo     matplotlib     -- PNG polar / 1D plot generation
echo     plotly         -- interactive 3D farfield HTML
echo.
echo   NOT installed by this script (install separately):
echo     Python 3.10+   -- https://www.python.org/downloads/
echo     LibreVNA-GUI   -- https://github.com/jankae/LibreVNA/releases
echo                       (version 1.6.0 or newer required)
echo     MKS CANable    -- USB driver from MKS or SLCAN driver
echo  =====================================================
echo.
pause
exit /b 0

:error
echo.
echo  [ERROR] Installation failed at the step above.
echo          Check your internet connection and try again.
echo          If the problem persists, run pip manually:
echo            pip install ^<package-name^>
echo.
pause
exit /b 1
