"""Shared fixtures: every test gets its own KARL home on disk."""

from __future__ import annotations

import pytest


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("KARL_HOME", str(tmp_path))
    for var in ("KARL_BASE_URL", "KARL_MODEL", "KARL_WORKSPACE",
                "KARL_SHELL", "KARL_WEB", "KARL_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    from karl.config import Project
    return Project("test").ensure()
