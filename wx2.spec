# -*- mode: python ; coding: utf-8 -*-
"""
wx2.spec — Empaqueta wx2.py en modo onedir
Incluye todas las dependencias dinámicas necesarias
"""

import os
import shutil
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
)

# Módulos ocultos que se usan dinámicamente
hidden_imports = (
    collect_submodules("torch") +
    collect_submodules("torchaudio") +
    collect_submodules("pyannote.audio") +
    collect_submodules("transformers") +
    collect_submodules("rich") +  # ← Logger colorido
    ["_socket"]                   # ← Evita el error de runtime por pkg_resources
)

# Archivos de datos necesarios (incluso .py de los módulos)
datas = (
    collect_data_files("torch", include_py_files=True) +
    collect_data_files("torchaudio", include_py_files=True) +
    collect_data_files("pyannote.audio", include_py_files=True) +
    collect_data_files("transformers", include_py_files=True) +
    collect_data_files("rich", include_py_files=True)
)

block_cipher = None

# Análisis del script principal
a = Analysis(
    ["wx2.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

# Empaquetado de scripts Python
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Ejecutable principal
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=False,
    name="wx2_temp",         # Nombre temporal
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,            # True para ver logs en consola
)

# Colección de archivos en dist/wx2
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="wx2"
)

# Renombrar el ejecutable final
final_dir = os.path.join("dist", "wx2")
old_exe = os.path.join(final_dir, "wx2_temp.exe")
new_exe = os.path.join(final_dir, "wx2.exe")

if os.path.exists(old_exe):
    shutil.move(old_exe, new_exe)
