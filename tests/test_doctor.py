"""Checks that device selection and failures cannot silently claim success."""

import pytest

from dating_compat import doctor
from dating_compat.__main__ import main


def test_auto_prefers_available_accelerator():
    assert doctor.select_device("auto", {"cpu": True, "mps": True, "cuda": False}) == "mps"
    assert doctor.select_device("auto", {"cpu": True, "mps": False, "cuda": False}) == "cpu"


def test_explicit_unavailable_device_is_rejected():
    with pytest.raises(ValueError, match="unavailable"):
        doctor.select_device("cuda", {"cpu": True, "mps": True, "cuda": False})


def test_missing_dependency_fails_diagnosis(monkeypatch):
    original = doctor.importlib.import_module

    def missing(name):
        if name == "hdbscan":
            raise ImportError("simulated missing dependency")
        return original(name)

    monkeypatch.setattr(doctor.importlib, "import_module", missing)
    report = doctor.diagnose("cpu")
    assert not report["passed"]
    assert any("simulated missing dependency" in error for error in report["errors"])


def test_failed_cli_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr("dating_compat.__main__.diagnose", lambda _: {"passed": False})
    assert main(["doctor"]) == 1
    assert "ENVIRONMENT FAIL" in capsys.readouterr().out
