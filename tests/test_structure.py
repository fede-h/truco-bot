"""Test that all packages and modules in truco_bot can be imported cleanly."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "truco_bot",
        "truco_bot.core",
        "truco_bot.core.card",
        "truco_bot.core.deck",
        "truco_bot.core.state",
        "truco_bot.core.actions",
        "truco_bot.core.rules",
        "truco_bot.env",
        "truco_bot.env.engine",
        "truco_bot.env.masking",
        "truco_bot.env.obs",
        "truco_bot.agents",
        "truco_bot.agents.baselines",
        "truco_bot.agents.cfr",
        "truco_bot.eval",
        "truco_bot.eval.arena",
    ],
)
def test_import_modules(module_name: str):
    module = importlib.import_module(module_name)
    assert module is not None
