"""
Unit tests for tools/symbolic_math.py
"""
import pytest
import numpy as np
import sympy as sp

from tools.symbolic_math import (
    check_structural_identifiability,
    evaluate_sensitivities,
    find_lumped_parameters,
    make_ode_simulator,
    reparameterize_ode,
)

# ── True pendulum values used in assertions ───────────────────────────────────
J, m_val, g_val, L_val, b_v = 0.05, 0.5, 9.81, 0.30, 0.02
K_G_TRUE = m_val * g_val * L_val / J    # 29.43
TAU_D_TRUE = b_v / J                    # 0.40
K_IN_TRUE = 1.0 / J                     # 20.0

PENDULUM_RHS    = "tau/J - b_v/J*theta_dot - m*g*L/J*sin(theta)"
PENDULUM_PARAMS = ["J", "b_v", "m", "g", "L"]
STATE_VARS      = ["theta", "theta_dot"]
INPUT_VARS      = ["tau"]


# ── find_lumped_parameters ────────────────────────────────────────────────────

class TestFindLumpedParameters:
    def test_linear_term_extracted(self):
        result = find_lumped_parameters(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS
        )
        # theta_dot coefficient should be -b_v/J
        assert "coeff_theta_dot" in result

    def test_input_term_extracted(self):
        result = find_lumped_parameters(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS
        )
        assert "coeff_tau" in result

    def test_sin_term_extracted(self):
        result = find_lumped_parameters(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS
        )
        assert "coeff_sin_theta" in result

    def test_three_combinations_found(self):
        result = find_lumped_parameters(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS
        )
        assert len(result) == 3

    def test_simple_linear_ode(self):
        # ẋ = -a*x + b*u  →  coeff_x = -a, coeff_u = b
        result = find_lumped_parameters(
            "-a*x + b*u", ["a", "b"], ["x"], ["u"]
        )
        assert "coeff_x" in result
        assert "coeff_u" in result

    def test_bad_ode_raises(self):
        with pytest.raises(ValueError):
            find_lumped_parameters("ZZZZ!@#", ["a"], ["x"], ["u"])


# ── check_structural_identifiability ─────────────────────────────────────────

class TestCheckStructuralIdentifiability:
    def _run(self, lnames=None):
        return check_structural_identifiability(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS,
            lumped_names=lnames,
        )

    def test_status_is_none_or_partial(self):
        # Individual J, b_v, m, g, L are not identifiable from angle alone
        result = self._run()
        assert result["identifiable"] in ("none", "partial")

    def test_returns_non_identifiable_list(self):
        result = self._run()
        # At least some params should be non-identifiable
        assert len(result["non_identifiable_params"]) > 0

    def test_lumped_params_present(self):
        result = self._run()
        assert len(result["lumped_params"]) >= 2

    def test_human_readable_names_applied(self):
        names = {
            "coeff_tau":         "K_in",
            "coeff_theta_dot":   "tau_d",
            "coeff_sin_theta":   "K_g",
        }
        result = self._run(lnames=names)
        assert "K_in" in result["lumped_params"]
        assert "tau_d" in result["lumped_params"]
        assert "K_g" in result["lumped_params"]

    def test_recommendation_non_empty(self):
        result = self._run()
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    def test_fully_identifiable_model(self):
        # ẋ = -a*x + b*u  with a and b individually identifiable
        r = check_structural_identifiability(
            "-a*x + b*u", ["a", "b"], ["x"], ["u"]
        )
        assert r["identifiable"] == "full"
        assert r["non_identifiable_params"] == []


# ── reparameterize_ode ────────────────────────────────────────────────────────

class TestReparameterizeOde:
    def test_produces_lumped_symbols(self):
        subs = {"K_g": "m*g*L/J", "tau_d": "b_v/J", "K_in": "1/J"}
        result = reparameterize_ode(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS, subs
        )
        # Result should contain the new symbols and not the original ones
        assert "K_g" in result or "tau_d" in result or "K_in" in result

    def test_numerical_equivalence(self):
        """Reparameterized expression should evaluate to the same number."""
        import sympy as sp

        subs = {"K_g": "m*g*L/J", "tau_d": "b_v/J", "K_in": "1/J"}
        reparam = reparameterize_ode(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS, subs
        )

        # Evaluate original
        vals_orig = {"J": J, "b_v": b_v, "m": m_val, "g": g_val, "L": L_val,
                     "theta_dot": 0.5, "theta": 0.3, "tau": 1.0}
        orig_expr = sp.sympify(PENDULUM_RHS, locals={k: sp.Symbol(k) for k in vals_orig})
        orig_val = float(orig_expr.subs(vals_orig))

        # Evaluate reparameterized
        vals_lump = {"K_g": K_G_TRUE, "tau_d": TAU_D_TRUE, "K_in": K_IN_TRUE,
                     "theta_dot": 0.5, "theta": 0.3, "tau": 1.0}
        reparam_expr = sp.sympify(reparam, locals={k: sp.Symbol(k) for k in vals_lump})
        reparam_val = float(reparam_expr.subs(vals_lump))

        assert abs(orig_val - reparam_val) < 1e-6


# ── make_ode_simulator ────────────────────────────────────────────────────────

class TestMakeOdeSimulator:
    def _get_simulator(self):
        rhs = "K_in*tau - tau_d*theta_dot - K_g*sin(theta)"
        return make_ode_simulator(
            rhs,
            fit_params=["K_in", "tau_d", "K_g"],
            state_vars=["theta", "theta_dot"],
            input_vars=["tau"],
            highest_deriv_var="theta_ddot",
        )

    def test_returns_callable(self):
        sim = self._get_simulator()
        assert callable(sim)

    def test_output_shape(self):
        sim = self._get_simulator()
        t = np.linspace(0, 2, 100)
        u = np.zeros(100)
        params = np.array([K_IN_TRUE, TAU_D_TRUE, K_G_TRUE])
        y = sim(params, t, u, x0=np.array([0.3, 0.0]))
        assert y.shape == (100,)

    def test_free_oscillation_decays(self):
        """Free oscillation (u=0) should decay due to damping."""
        sim = self._get_simulator()
        t = np.linspace(0, 10, 500)
        u = np.zeros(500)
        params = np.array([K_IN_TRUE, TAU_D_TRUE, K_G_TRUE])
        y = sim(params, t, u, x0=np.array([0.3, 0.0]))
        # Amplitude at the end should be less than at the start
        assert abs(y[-1]) < abs(y[0])

    def test_zero_initial_condition(self):
        """With no torque and zero IC, output should be near zero."""
        sim = self._get_simulator()
        t = np.linspace(0, 1, 50)
        u = np.zeros(50)
        params = np.array([K_IN_TRUE, TAU_D_TRUE, K_G_TRUE])
        y = sim(params, t, u, x0=np.array([0.0, 0.0]))
        assert np.all(np.abs(y) < 1e-4)

    def test_responds_to_input(self):
        """Constant torque should produce nonzero angle response."""
        sim = self._get_simulator()
        t = np.linspace(0, 2, 200)
        u = np.full(200, 0.5)
        params = np.array([K_IN_TRUE, TAU_D_TRUE, K_G_TRUE])
        y = sim(params, t, u, x0=np.array([0.0, 0.0]))
        assert np.max(np.abs(y)) > 0.01


# ── evaluate_sensitivities ────────────────────────────────────────────────────

class TestEvaluateSensitivities:
    def test_returns_dict_for_each_param(self):
        result = evaluate_sensitivities(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS
        )
        for p in PENDULUM_PARAMS:
            assert p in result

    def test_sensitivity_expressions_are_strings(self):
        result = evaluate_sensitivities(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS
        )
        for v in result.values():
            assert isinstance(v, str)

    def test_sensitivity_wrt_g_contains_expected_terms(self):
        result = evaluate_sensitivities(
            PENDULUM_RHS, PENDULUM_PARAMS, STATE_VARS, INPUT_VARS
        )
        # ∂rhs/∂g = -m*L/J * sin(theta)  → should contain L and m
        sens_g = result["g"]
        assert "L" in sens_g or "m" in sens_g
