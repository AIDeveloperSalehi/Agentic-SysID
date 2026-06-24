"""
Unit tests for tools/experiment_design_toolkit.py
"""
import numpy as np
import pytest

from core.schemas import PlantContract
from tools.experiment_design_toolkit import (
    compute_information_content,
    design_adversarial_inputs,
    design_compound_identification_input,
    design_identification_input,
)


@pytest.fixture
def pendulum_contract():
    return PlantContract(
        name="pendulum",
        input_names=["torque"],
        output_names=["angle"],
        input_limits={"torque": (-2.0, 2.0)},
        sample_time=0.02,
    )


# ── design_identification_input ───────────────────────────────────────────────

class TestDesignIdentificationInput:
    def test_prbs_shape(self, pendulum_contract):
        r = design_identification_input(pendulum_contract, n_samples=200, method="prbs")
        assert r["u"].shape == (200,)
        assert r["t"].shape == (200,)

    def test_multisine_shape(self, pendulum_contract):
        r = design_identification_input(pendulum_contract, n_samples=200, method="multisine")
        assert r["u"].shape == (200,)

    def test_chirp_shape(self, pendulum_contract):
        r = design_identification_input(pendulum_contract, n_samples=200, method="chirp")
        assert r["u"].shape == (200,)

    def test_steps_returns_data(self, pendulum_contract):
        r = design_identification_input(pendulum_contract, n_samples=200, method="steps")
        assert len(r["u"]) > 0

    def test_amplitude_within_limits(self, pendulum_contract):
        for method in ["prbs", "multisine", "chirp"]:
            r = design_identification_input(pendulum_contract, n_samples=200, method=method)
            assert np.all(r["u"] >= -2.0), f"{method}: below lower limit"
            assert np.all(r["u"] <= 2.0),  f"{method}: above upper limit"

    def test_unknown_method_raises(self, pendulum_contract):
        with pytest.raises(ValueError):
            design_identification_input(pendulum_contract, method="unknown")

    def test_reproducible_with_seed(self, pendulum_contract):
        r1 = design_identification_input(pendulum_contract, seed=7)
        r2 = design_identification_input(pendulum_contract, seed=7)
        np.testing.assert_array_equal(r1["u"], r2["u"])

    def test_different_seeds_differ(self, pendulum_contract):
        r1 = design_identification_input(pendulum_contract, seed=1)
        r2 = design_identification_input(pendulum_contract, seed=2)
        assert not np.allclose(r1["u"], r2["u"])

    def test_metadata_fields_present(self, pendulum_contract):
        r = design_identification_input(pendulum_contract)
        for key in ("t", "u", "method", "description", "input_type"):
            assert key in r


# ── design_compound_identification_input ──────────────────────────────────────

class TestDesignCompoundInput:
    def test_correct_length(self, pendulum_contract):
        r = design_compound_identification_input(pendulum_contract, n_samples=400)
        assert len(r["u"]) == 400

    def test_within_limits(self, pendulum_contract):
        r = design_compound_identification_input(pendulum_contract, n_samples=400)
        assert np.all(r["u"] >= -2.0)
        assert np.all(r["u"] <= 2.0)


# ── design_adversarial_inputs ─────────────────────────────────────────────────

class TestDesignAdversarialInputs:
    def test_returns_n_scenarios(self, pendulum_contract):
        scenarios = design_adversarial_inputs(pendulum_contract, n_scenarios=3)
        assert len(scenarios) == 3

    def test_each_scenario_has_required_fields(self, pendulum_contract):
        for sc in design_adversarial_inputs(pendulum_contract, n_scenarios=3):
            for key in ("t", "u", "description", "scenario_type"):
                assert key in sc, f"Missing '{key}' in scenario {sc.get('scenario_type')}"

    def test_inputs_within_limits(self, pendulum_contract):
        for sc in design_adversarial_inputs(pendulum_contract, n_scenarios=3):
            assert np.all(sc["u"] >= -2.0), f"Below limit in {sc['scenario_type']}"
            assert np.all(sc["u"] <= 2.0),  f"Above limit in {sc['scenario_type']}"

    def test_slow_sinusoidal_is_first(self, pendulum_contract):
        scenarios = design_adversarial_inputs(pendulum_contract, n_scenarios=3)
        assert scenarios[0]["scenario_type"] == "low_freq_sine"

    def test_fewer_scenarios_than_max(self, pendulum_contract):
        scenarios = design_adversarial_inputs(pendulum_contract, n_scenarios=2)
        assert len(scenarios) == 2

    def test_different_seeds_differ(self, pendulum_contract):
        s1 = design_adversarial_inputs(pendulum_contract, seed=0)
        s2 = design_adversarial_inputs(pendulum_contract, seed=99)
        # large_amplitude PRBS (index 1) changes with seed
        assert not np.allclose(s1[1]["u"], s2[1]["u"])


# ── compute_information_content ───────────────────────────────────────────────

class TestComputeInformationContent:
    def _sensitivity_fn(self, u):
        # Fake: S[i, j] = sin(i*j*0.1) for 2 params
        N = len(u)
        S = np.column_stack([
            np.sin(np.arange(N) * 0.1) * u,
            np.cos(np.arange(N) * 0.1) * u,
        ])
        return S

    def test_returns_expected_keys(self, pendulum_contract):
        r = design_identification_input(pendulum_contract, n_samples=100, method="prbs")
        result = compute_information_content(r["t"], r["u"], self._sensitivity_fn)
        for key in ("logdet_fim", "fim_diagonal", "rank"):
            assert key in result

    def test_logdet_is_finite_for_rich_input(self, pendulum_contract):
        r = design_identification_input(pendulum_contract, n_samples=200, method="prbs")
        result = compute_information_content(r["t"], r["u"], self._sensitivity_fn)
        assert np.isfinite(result["logdet_fim"])

    def test_zero_input_gives_low_information(self):
        t = np.linspace(0, 2, 100)
        u = np.zeros(100)
        result = compute_information_content(t, u, self._sensitivity_fn)
        assert result["logdet_fim"] <= 0.0
