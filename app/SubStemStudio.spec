import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


root = Path(SPECPATH).parent
analysis = Analysis(
    [str(root / "app" / "launcher.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / "subroformer" / "configs"), "subroformer/configs"),
           (str(root / "app" / "assets"), "app/assets"),
           (str(root / "subroformer" / "frontend" / "subband" / "dsp2" / "csrc"), "subroformer/frontend/subband/dsp2/csrc"),
           (str(root / "subroformer" / "frontend" / "subband" / "dsp3" / "csrc"), "subroformer/frontend/subband/dsp3/csrc")],
    hiddenimports=collect_submodules("librosa"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "pytest"],
    noarchive=False,
    module_collection_mode={"librosa": "pyz+py", "torch": "pyz+py"},
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="SubStemStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(root / "app" / "assets" / "subroformer-icon.ico") if sys.platform == "win32" else None,
)
collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="SubStemStudio",
)
if sys.platform == "darwin":
    bundle = BUNDLE(
        collection,
        name="SubStemStudio.app",
        icon=str(root / "app" / "assets" / "subroformer-icon.icns"),
        bundle_identifier="com.archimickey.subroformer",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "LSMinimumSystemVersion": "15.0",
            "NSHighResolutionCapable": True,
        },
    )
