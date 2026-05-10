from types import SimpleNamespace

import pytest

import tests.helpers.run_sh_command as run_sh_command_module


class _FakeErrorReturnCode(Exception):
    def __init__(self) -> None:
        self.stderr = b"simulated command failure"


def test_run_sh_command_reports_stderr_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_error(command: list[str]) -> None:
        raise _FakeErrorReturnCode()

    fake_sh = SimpleNamespace(python=_raise_error, ErrorReturnCode=_FakeErrorReturnCode)
    monkeypatch.setattr(run_sh_command_module, "sh", fake_sh, raising=False)

    with pytest.raises(pytest.fail.Exception, match="simulated command failure"):
        run_sh_command_module.run_sh_command(["script.py"])
