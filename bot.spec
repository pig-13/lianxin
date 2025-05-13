# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['lianxin_launcher.py'],  # ✅ 主程式改為 launcher
    pathex=[],
    binaries=[],
    datas=[
        ('bot.py', '.'),                # ✅ 要用到的 bot 檔
        ('memory_ui.py', '.'),          # ✅ Flask UI
        ('db_utils.py', '.'),           # ✅ 若有分 utils
        ('templates/', 'templates'),    # ✅ Flask templates 資料夾
        ('.env', '.'),                  # ✅ 若有預設的環境變數
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='lianxin_launcher',  # ✅ 這樣會產生 lianxin_launcher.exe
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # ✅ 如果你想要有 log 輸出視窗
    icon='lianxin.ico'  # ✅ 若有設定圖示
)
