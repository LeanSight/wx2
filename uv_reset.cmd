@echo off
REM Script to reset the uv environment on Windows

echo Deleting the virtual environment (.venv)...
if exist .venv (
    rmdir /s /q .venv
    echo .venv has been deleted.
) else (
    echo .venv does not exist.
)

echo Deleting uv.lock...
if exist uv.lock (
    del /f /q uv.lock
    echo uv.lock has been deleted.
) else (
    echo uv.lock does not exist.
)

echo Deleting uv.cache directory...
if exist uv.cache (
    rmdir /s /q uv.cache
    echo uv.cache has been deleted.
) else (
    echo uv.cache does not exist.
)

echo Cleaning the global uv cache...
uv cache clean
