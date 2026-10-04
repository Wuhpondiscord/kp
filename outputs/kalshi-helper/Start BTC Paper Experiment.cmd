@echo off
setlocal
cd /d "%~dp0"
echo BTC prospective paper experiment - simulated money only.
echo Keep this terminal and computer awake. Press Ctrl+C to stop.
echo Running this launcher again resumes the same frozen experiment.
if exist "data\btc-forward\protocol.json" goto record
python -m helper.btc_forward init --directory data/btc-forward
if errorlevel 1 goto failed
:record
python -m helper.btc_forward record --directory data/btc-forward --polls 172800 --interval 15
if errorlevel 1 goto failed
python -m helper.btc_forward settle --directory data/btc-forward --polls 20 --interval 15
if errorlevel 1 goto failed
python -m helper.btc_forward replay --directory data/btc-forward
if errorlevel 1 goto failed
echo Results saved to data\btc-forward\replay.json
pause
exit /b 0
:failed
echo Experiment stopped. Review the error above; do not modify the frozen run to bypass it.
pause
exit /b 1
