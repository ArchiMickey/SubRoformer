"""Run with the build environment's Python on the target operating system."""

import subprocess
import sys
import tempfile
from pathlib import Path


def build_disk_image(directory):
    distribution = directory / "dist"
    with tempfile.TemporaryDirectory(prefix="subroformer-dmg-") as temporary:
        staging = Path(temporary)
        subprocess.run(
            ["ditto", str(distribution / "SubRoformer.app"), str(staging / "SubRoformer.app")],
            check=True,
        )
        (staging / "Applications").symlink_to("/Applications", target_is_directory=True)
        subprocess.run(
            ["hdiutil", "create", "-ov", "-volname", "SubRoformer", "-fs", "HFS+",
             "-format", "UDZO", "-srcfolder", str(staging), str(distribution / "SubRoformer.dmg")],
            check=True,
        )


def main():
    directory = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm",
         "--distpath", str(directory / "dist"),
         "--workpath", str(directory / "build"),
         str(directory / "SubRoformer.spec")],
        cwd=directory.parent,
        check=True,
    )
    if sys.platform == "darwin":
        build_disk_image(directory)


if __name__ == "__main__":
    main()
