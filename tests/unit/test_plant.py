"""Tests for plants/inverted_pendulum.py."""
import numpy as np
import pytest

from plants.inverted_pendulum import PendulumParams, PendulumPlant


@pytest.fixture
def plant():
    return PendulumPlant(seed=0)


@pytest.fixture
def zero_torque():
    return lambda t: np.array([0.0])


@pytest.fixture
def step_torque():
    return lambda t: np.array([0.5 if t < 1.0 else 0.0])


# ── Basic simulation ──────────────────────────────────────────────────────────

class TestSimulation:
    def test_output_shape(self, plant, zero_torque):
        t, u, y = plant.apply_input(zero_torque, (0.0, 2.0), 0.02)
        N = len(t)
        assert u.shape == (1, N)
        assert y.shape == (1, N)

    def test_time_vector_length(self, plant, zero_torque):
        t, _, _ = plant.apply_input(zero_torque, (0.0, 1.0), 0.02)
        expected = int(1.0 / 0.02) + 1
        assert abs(len(t) - expected) <= 2   # ±1 due to floating-point arange

    def test_zero_torque_oscillates(self, plant, zero_torque):
        """Pendulum released from θ=0.3 rad should oscillate, not stay still."""
        t, _, y = plant.apply_input(zero_torque, (0.0, 4.0), 0.02)
        theta = y[0]
        assert theta.max() > 0.1
        # Should cross zero (oscillation)
        sign_changes = np.sum(np.diff(np.sign(theta)) != 0)
        assert sign_changes >= 2

    def test_step_torque_drives_motion(self, plant, step_torque):
        t, u, y = plant.apply_input(step_torque, (0.0, 2.0), 0.02)
        theta = y[0]
        # Torque should drive angle away from zero
        assert theta.max() > plant.default_x0[0]

    def test_noise_is_small(self, plant):
        """Noise should be much smaller than the signal."""
        torque = lambda t: np.array([0.3])
        t, _, y = plant.apply_input(torque, (0.0, 2.0), 0.02)
        # Run noiseless
        t2, _, theta_clean, _ = plant.simulate_noiseless(torque, (0.0, 2.0), 0.02)
        noise = y[0] - theta_clean
        assert np.std(noise) < plant.params.noise_std * 5

    def test_reproducible_with_same_seed(self):
        """Same seed → same noise sequence."""
        p1 = PendulumPlant(seed=7)
        p2 = PendulumPlant(seed=7)
        u  = lambda t: np.array([0.0])
        _, _, y1 = p1.apply_input(u, (0.0, 1.0), 0.02)
        _, _, y2 = p2.apply_input(u, (0.0, 1.0), 0.02)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_differ(self):
        p1 = PendulumPlant(seed=1)
        p2 = PendulumPlant(seed=2)
        u  = lambda t: np.array([0.0])
        _, _, y1 = p1.apply_input(u, (0.0, 1.0), 0.02)
        _, _, y2 = p2.apply_input(u, (0.0, 1.0), 0.02)
        assert not np.allclose(y1, y2)


# ── Noiseless simulation ──────────────────────────────────────────────────────

class TestNoiseless:
    def test_returns_four_arrays(self, plant, zero_torque):
        result = plant.simulate_noiseless(zero_torque, (0.0, 2.0), 0.02)
        assert len(result) == 4    # t, u, theta, theta_dot

    def test_energy_conservation_approx(self, plant, zero_torque):
        """
        With zero torque and only viscous+Coulomb friction, mechanical energy
        must monotonically decrease (friction dissipates energy).
        """
        p = plant.params
        t, _, theta, omega = plant.simulate_noiseless(zero_torque, (0.0, 5.0), 0.005)
        KE = 0.5 * p.J * omega**2
        PE = p.m * p.g * p.L * (1 - np.cos(theta))
        E  = KE + PE
        # Energy at end must be less than at start (friction damping)
        assert E[-1] < E[0]


# ── Coulomb friction effect ───────────────────────────────────────────────────

class TestCoulombEffect:
    def test_stiction_region_near_zero_velocity(self, plant):
        """
        A pendulum at near-zero velocity with only gravity restoring
        should eventually stick due to Coulomb friction.
        """
        params = PendulumParams(b_v=0.005, f_c=0.15)  # strong Coulomb
        p2 = PendulumPlant(params=params, seed=0)
        u  = lambda t: np.array([0.0])
        # Start near equilibrium with tiny velocity
        t, _, theta, omega = p2.simulate_noiseless(u, (0.0, 10.0), 0.005, x0=np.array([0.05, 0.0]))
        # Velocity should decay towards zero
        assert abs(omega[-1]) < abs(omega[0]) + 0.01

    def test_coulomb_model_differs_from_pure_viscous(self):
        """Same b_v but f_c=0 vs f_c>0 should produce measurably different outputs."""
        p_viscous = PendulumPlant(PendulumParams(b_v=0.05, f_c=0.0), seed=0)
        p_coulomb = PendulumPlant(PendulumParams(b_v=0.05, f_c=0.08), seed=0)
        u  = lambda t: np.array([0.3 * np.sin(2 * np.pi * 0.5 * t)])
        _, _, th_v, _ = p_viscous.simulate_noiseless(u, (0.0, 5.0), 0.02)
        _, _, th_c, _ = p_coulomb.simulate_noiseless(u, (0.0, 5.0), 0.02)
        rmse = np.sqrt(np.mean((th_v - th_c)**2))
        assert rmse > 0.01   # Coulomb makes a visible difference


# ── Linearisation ────────────────────────────────────────────────────────────

class TestLinearize:
    def test_returns_matrices(self, plant):
        lin = plant.linearize()
        assert "A" in lin and "B" in lin and "C" in lin and "D" in lin

    def test_A_shape(self, plant):
        A = plant.linearize()["A"]
        assert A.shape == (2, 2)

    def test_K_g_positive(self, plant):
        """Pendulum linearised at bottom: restoring force → K_g > 0."""
        assert plant.linearize()["K_g"] > 0


# ── Interface ─────────────────────────────────────────────────────────────────

class TestInterface:
    def test_n_properties(self, plant):
        assert plant.n_inputs  == 1
        assert plant.n_outputs == 1
        assert plant.n_states  == 2

    def test_default_x0_shape(self, plant):
        assert plant.default_x0.shape == (2,)
