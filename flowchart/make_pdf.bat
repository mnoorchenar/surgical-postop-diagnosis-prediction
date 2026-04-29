@echo off
setlocal EnableDelayedExpansion
title Make Flowchart PDF
cd /d "%~dp0"
echo.
echo  flowchart.html  --^>  flowchart.svg  +  flowchart.pdf
echo.

:: ── Step 1: extract SVG from HTML ──────────────────────────────────────────
python -c "import re,pathlib;d=pathlib.Path('.');t=d.joinpath('flowchart.html').read_text(encoding='utf-8');m=re.search(r'(<svg[\s\S]*?</svg>)',t);d.joinpath('flowchart.svg').write_text(m.group(1),encoding='utf-8');print('  [1/2] flowchart.svg written (%d bytes)' %% d.joinpath('flowchart.svg').stat().st_size)"
if errorlevel 1 goto :fail

:: ── Step 2: HTML -> PDF via Edge (full font support, exact browser rendering)
set EDGE=
for %%E in (
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
) do if not defined EDGE (if exist %%E set EDGE=%%E)

if defined EDGE (
    echo  [2/2] Using Microsoft Edge for PDF ...
    :: Build absolute file:/// URL via Python
    for /f "delims=" %%U in ('python -c "import pathlib;print(pathlib.Path('flowchart.html').resolve().as_uri())"') do set HTMLURL=%%U
    for /f "delims=" %%P in ('python -c "import pathlib;print(str(pathlib.Path('flowchart.pdf').resolve()))"') do set PDFPATH=%%P

    %EDGE% --headless=new --disable-gpu --no-pdf-header-footer ^
        "--print-to-pdf=!PDFPATH!" "!HTMLURL!" >nul 2>&1

    if exist "flowchart.pdf" (
        for %%F in (flowchart.pdf) do echo         Done^^!  flowchart.pdf  ^(%%~zF bytes^)
        goto :done
    )
    echo         Edge produced no output. Falling back to cairosvg...
)

:: ── Step 2b: fallback — cairosvg (limited Unicode font support) ─────────────
echo  [2/2] Using cairosvg for PDF ...
python -c "import pathlib;d=pathlib.Path('.');__import__('cairosvg').svg2pdf(url=str(d.joinpath('flowchart.svg').resolve()),write_to=str(d.joinpath('flowchart.pdf')));print('  Done^^!  flowchart.pdf  (%d bytes)' %% d.joinpath('flowchart.pdf').stat().st_size)"
if errorlevel 1 goto :fail
goto :done

:fail
echo.
echo  FAILED — check errors above.
echo  Ensure cairosvg is installed:  pip install cairosvg
echo.
pause
exit /b 1

:done
echo.
pause
