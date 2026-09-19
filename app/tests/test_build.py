import ast
from pathlib import Path

import pytest

from app import build


def test_bundle_includes_dsp3_native_sources():
    root = Path(__file__).resolve().parents[2]
    spec = ast.parse((root / "app" / "SubStemStudio.spec").read_text())
    constants = [node.value for node in ast.walk(spec) if isinstance(node, ast.Constant)]
    assert "dsp3/csrc" in constants
    for name in ("shift_frequency_cpu.cpp", "shift_frequency_mps.metal"):
        assert (root / "dsp3" / "csrc" / name).is_file()


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_build_creates_disk_image_only_on_macos(monkeypatch, platform):
    commands = []
    images = []
    monkeypatch.setattr(build.sys, "platform", platform)
    monkeypatch.setattr(build.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(build, "build_disk_image", images.append)
    build.main()
    assert "PyInstaller" in commands[0]
    assert Path(commands[0][-1]).name == "SubStemStudio.spec"
    assert len(images) == (1 if platform == "darwin" else 0)


def test_disk_image_contains_app_and_applications_link(monkeypatch, tmp_path):
    commands = []

    def run(command, **kwargs):
        assert kwargs["check"]
        commands.append(command)
        if command[0] == "hdiutil":
            staging = Path(command[command.index("-srcfolder") + 1])
            assert (staging / "Applications").readlink() == Path("/Applications")

    monkeypatch.setattr(build.subprocess, "run", run)
    build.build_disk_image(tmp_path)
    assert commands[0][0] == "ditto"
    assert commands[0][1] == str(tmp_path / "dist" / "SubStemStudio.app")
    assert commands[1][-1] == str(tmp_path / "dist" / "SubStemStudio.dmg")
    assert "UDZO" in commands[1]
    assert commands[1][commands[1].index("-volname") + 1] == "SubStemStudio"
    assert Path(commands[0][2]).name == "SubStemStudio.app"
    assert not Path(commands[0][2]).parent.exists()
