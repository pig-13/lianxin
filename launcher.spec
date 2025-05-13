# lianxin_launcher.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['lianxin_launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('bot.py', '.'), 
        ('memory_ui.py', '.'), 
        ('lianxin.ico', '.')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='lianxin_launcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # ❗視情況改成 False
    icon='lianxin.ico'
)
