This should be README.md

# Python Code Documentation

This document provides an overview of the Python code developed for
controlling a pendulum system using optimal control theory.

## Functions Overview

-   **ham(vec)**: Calculates the Hamiltonian for the system given state
    and costate variables.
-   **find_cost(t, vect)**: Computes the total cost of a trajectory at a
    given time.
-   **f(t, y)**: Defines the differential equations governing the
    dynamics of the pendulum and its costates.
-   **find_closest_state(P, theta0, phi0, prev_angle, angle, res)**:
    Iteratively finds the closest state from a specified range of
    angles.
-   **control_pendulum(theta0, phi0, alpha)**: Sets up and controls the
    pendulum to reach a resting state, minimizing the cost function.

## Usage

The code can be used to study and optimize the control of a pendulum
with or without external forces and damping. Here is a brief example of
how to use the functions:

            theta0 = 0.1  # Initial angle
            phi0 = -0.02  # Initial angular velocity
            alpha = 0.1   # Damping coefficient
            result = control_pendulum(theta0, phi0, alpha)
            print("Optimal costates and cost:", result)
        

## Dependencies

This code requires the following Python libraries:

-   NumPy
-   SciPy
-   Math

## Notes

Ensure all dependencies are installed and Python version used is
compatible with the libraries. This code is designed for educational and
research purposes and may require adjustments for commercial use.
