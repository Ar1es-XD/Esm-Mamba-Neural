@echo off
REM ============================================================================
REM  run_pipeline.bat — ESM-Mamba Neural Network Experiments
REM  Run this on the rig (Ryzen 9 9900X / RTX PRO 4000 Blackwell).
REM ============================================================================
setlocal

echo =====================================================================
echo  ESM-Mamba (MambaCross) Neural Network Pipeline
echo  Target: RTX PRO 4000 (24 GB VRAM) / Ryzen 9 9900X / 64 GB DDR5
echo =====================================================================
echo.

REM ── Activate virtual environment ──────────────────────────────────────
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found. Run:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    exit /b 1
)

REM ── CUDA sanity check ─────────────────────────────────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'No CUDA — check NVIDIA drivers and PyTorch CUDA build'"
if errorlevel 1 (
    echo.
    echo FATAL: CUDA is not available. This pipeline requires an NVIDIA GPU.
    exit /b 1
)
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB'); print(f'  PyTorch: {torch.__version__}')"
echo.

REM ── Phase A: ESM-2 Embedding Extraction (skip if already done) ───────
if exist "Outputs\Pretrained_HIV\ab" (
    echo [Phase A] ESM-2 embeddings already exist — skipping extraction.
) else (
    echo [Phase A] Extracting ESM-2 embeddings...
    python shared\Pretrained.py
    if errorlevel 1 (
        echo ERROR: ESM-2 extraction failed.
        exit /b 1
    )
)
echo.

REM ── Phase B: Run all 4 experiments ────────────────────────────────────
echo [Phase B] Running all 4 neural network experiments (30 epochs each)...
echo.
python run_all_experiments.py
if errorlevel 1 (
    echo ERROR: Experiment pipeline failed.
    exit /b 1
)

echo.
echo =====================================================================
echo  Pipeline complete. Results are in each experiment_*/results/ folder
echo  and consolidated in nn_summary_results.csv
echo =====================================================================
endlocal
