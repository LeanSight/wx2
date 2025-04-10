# -*- mode: python ; coding: utf-8 -*-
"""
wx2.spec — Packages wx2.py in onedir mode
Includes all necessary dynamic dependencies
"""

import os
import shutil
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
)

# Hidden modules that are used dynamically
hidden_imports = (
    collect_submodules("torch") +
    collect_submodules("torchaudio") +
    collect_submodules("pyannote.audio") +
    collect_submodules("transformers") +
    collect_submodules("rich") +  # ← Colorful logger
    ["_socket"]                   # ← Avoids runtime error from pkg_resources
)

# Necessary data files (including .py files from modules)
datas = (
    collect_data_files("torch", include_py_files=True) +
    collect_data_files("torchaudio", include_py_files=True) +
    collect_data_files("pyannote.audio", include_py_files=True) +
    collect_data_files("transformers", include_py_files=True) +
    collect_data_files("rich", include_py_files=True)
)

block_cipher = None

# Main script analysis
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

# Packaging Python scripts
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Main executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=False,
    name="wx2_temp",         # Temporary name
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,            # True to see logs in console
)

# Collection of files in dist/wx2
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

# Rename the final executable
final_dir = os.path.join("dist", "wx2")
old_exe = os.path.join(final_dir, "wx2_temp.exe")
new_exe = os.path.join(final_dir, "wx2.exe")

if os.path.exists(old_exe):
    shutil.move(old_exe, new_exe)