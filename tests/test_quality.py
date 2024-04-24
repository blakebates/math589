import sys
sys.path.append(".")
import numpy as np
import control_pendulum as cp
import driverlib as dl
dl.verbose=False

def test_quality():
    ob = dl.pendulum_components(α=.1, num_fronts=64, num_steps=360)
    ob.plot_fronts()
    # Test validity of the solution for theta0 in [0.1, 0.2], phi0 = 0."""
    θ0 = 0.1
    φ0 = 0
    cost_from_zero, Hval = ob.test_control_pendulum(cp.control_pendulum, θ0, φ0)
    print(f"The value of Hamiltonian of (theta0, phi0, lambda1, lambda2): {Hval}, should be 0")
    print(f"Successfully processed initial condition: theta0 = {θ0}, phi0 = {φ0}")
    max_cost_from_zero = 0.01
    print(f"Achieved cost from 0: { cost_from_zero }, max. cost allowed={max_cost_from_zero}")
    assert cost_from_zero < max_cost_from_zero

