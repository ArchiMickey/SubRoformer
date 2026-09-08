import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


root = Path(SPECPATH).parent
analysis = Analysis(
    [str(root / "app" / "launcher.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / "configs"), "configs"), (str(root / "app" / "assets"), "app/assets")],
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
    name="SubRoformer",
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
    name="SubRoformer",
)
if sys.platform == "darwin":
    bundle = BUNDLE(
        collection,
        name="SubRoformer.app",
        icon=str(root / "app" / "assets" / "subroformer-icon.icns"),
        bundle_identifier="com.archimickey.subroformer",
        info_plist={
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    )
