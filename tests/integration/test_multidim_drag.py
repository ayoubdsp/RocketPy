"""Integration tests for multi-dimensional drag coefficient support."""

import numpy as np

from rocketpy import Environment, Flight, Function, Rocket, SolidMotor


def test_flight_with_1d_drag(example_plain_env, calisto):
    """Test that flights with 1D drag curves still work (backward compatibility)."""

    flight = Flight(
        rocket=calisto,
        environment=example_plain_env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    # Check that flight completed successfully
    assert flight.t_final > 0
    assert flight.apogee > 0
    assert flight.apogee_time > 0


def test_flight_with_3d_drag_basic():
    """Test that a simple 3D drag function works."""
    # Create environment
    env = Environment(gravity=9.81)
    env.set_atmospheric_model(type="standard_atmosphere")

    # Create motor with simple constant thrust
    motor = SolidMotor(
        thrust_source=lambda t: 2000 if t < 3 else 0,
        burn_time=3.0,
        nozzle_radius=0.033,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        center_of_dry_mass_position=0.317,
        grains_center_of_mass_position=0.397,
        grain_number=5,
        grain_separation=0.005,
        grain_density=1815,
        grain_outer_radius=0.033,
        grain_initial_inner_radius=0.015,
        grain_initial_height=0.120,
        nozzle_position=0,
        throat_radius=0.011,
    )

    # Create 3D drag
    mach = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    reynolds = np.array([1e5, 5e5, 1e6])
    alpha = np.array([0.0, 2.0, 4.0, 6.0])

    M, Re, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    cd_data = 0.3 + 0.1 * M - 1e-7 * Re + 0.01 * A
    cd_data = np.clip(cd_data, 0.2, 1.0)

    power_off_drag = Function.from_grid(
        cd_data,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )
    power_on_drag = Function.from_grid(
        cd_data * 1.1,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )

    # Create rocket
    rocket = Rocket(
        radius=0.0635,
        mass=16.24,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=power_off_drag,
        power_on_drag=power_on_drag,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.set_rail_buttons(0.2, -0.5, 30)
    rocket.add_motor(motor, position=-1.255)

    # Run flight
    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    # Check results - should launch and have non-zero apogee
    assert flight.apogee > 100, f"Apogee too low: {flight.apogee}m"
    assert flight.apogee < 5000, f"Apogee too high: {flight.apogee}m"
    assert hasattr(flight, "angle_of_attack")


def test_3d_drag_with_varying_alpha():
    """Test that 3D drag responds to angle of attack changes."""
    # Create drag function with strong alpha dependency
    mach = np.array([0.0, 0.5, 1.0, 1.5])
    reynolds = np.array([1e5, 1e6])
    alpha = np.array([0.0, 5.0, 10.0, 15.0])

    M, Re, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    # Strong alpha dependency: Cd increases significantly with alpha
    cd_data = 0.3 + 0.05 * M + 0.03 * A
    cd_data = np.clip(cd_data, 0.2, 2.0)

    drag_func = Function.from_grid(
        cd_data,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )

    # Test at different angles of attack
    # At zero alpha, Cd should be lower
    cd_0 = drag_func(0.8, 5e5, 0.0)
    cd_10 = drag_func(0.8, 5e5, 10.0)

    # Cd should increase with alpha
    assert cd_10 > cd_0
    assert cd_10 - cd_0 > 0.2  # Should show significant difference
