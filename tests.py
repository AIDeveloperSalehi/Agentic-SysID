import numpy as np
import matplotlib.pyplot as plt
from plants.inverted_pendulum import PendulumPlant
from visualization.pendulum_viz import animate_pendulum, plot_phase_portrait

plant = PendulumPlant(seed=0)
u_func = lambda t: np.array([0.5 * np.sin(2 * np.pi * 0.5 * t)])
t, u, theta, omega = plant.simulate_noiseless(u_func, (0.0, 8.0), dt=0.02)

# Phase portrait
plot_phase_portrait([{"theta": theta, "theta_dot": omega, "label": "True plant"}])
plt.show()

# Animation (runs in a window)
anim = animate_pendulum(t, theta, u, L=0.30, title="Pendulum free response")
plt.show()