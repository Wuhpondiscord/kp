@echo off
cd /d "%~dp0"
if exist "..\..\work\browser-stage-data\research.sqlite3" (
  python -m helper --data "..\..\work\browser-stage-data" serve --port 8766
) else (
  python -m helper serve
)
pause
