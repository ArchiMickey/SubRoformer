import subprocess
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter


def main():
    assets = Path(__file__).resolve().parent / "assets"
    logo = QImage(str(assets / "subroformer-logo.png"))
    if logo.isNull():
        raise RuntimeError("Unable to load the original logo")
    icon = QImage(1024, 1024, QImage.Format.Format_ARGB32_Premultiplied)
    icon.fill(Qt.GlobalColor.transparent)
    painter = QPainter(icon)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.setBrush(QColor("#ffffff"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(72, 72, 880, 880), 196, 196)
    painter.drawImage(QRectF(112, 112, 800, 800), logo)
    painter.end()
    if not icon.save(str(assets / "subroformer-icon.png")):
        raise RuntimeError("Unable to save the app icon")
    if not icon.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation).save(str(assets / "subroformer-icon.ico")):
        raise RuntimeError("Unable to save the Windows icon")
    if sys.platform == "darwin":
        with tempfile.TemporaryDirectory(prefix="subroformer-icon-") as temporary:
            iconset = Path(temporary) / "SubRoformer.iconset"
            iconset.mkdir()
            for size in (16, 32, 128, 256, 512):
                for scale in (1, 2):
                    suffix = "@2x" if scale == 2 else ""
                    image = icon.scaled(size * scale, size * scale, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    if not image.save(str(iconset / f"icon_{size}x{size}{suffix}.png")):
                        raise RuntimeError("Unable to save an icon size")
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(assets / "subroformer-icon.icns")], check=True)


if __name__ == "__main__":
    main()
