"""Integration tests for multi-dimensional drag coefficient support."""

import numpy as np

from rocketpy import Flight, Function, Rocket


def test_flight_with_1d_drag(flight_calisto):
    """Test that flights with 1D drag curves still work (backward compatibility)."""

    # `flight_calisto` is a fixture that already runs the simulation
    flight = flight_calisto

    # Check that flight completed successfully
    assert flight.t_final > 0
    assert flight.apogee > 0
    assert flight.apogee_time > 0


def test_flight_with_3d_drag_basic(example_plain_env, cesaroni_m1670):
    """Test that a simple 3D drag function works."""
    # Use fixtures for environment and motor
    env = example_plain_env
    env.set_atmospheric_model(type="standard_atmosphere")
    motor = cesaroni_m1670

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


def test_3d_drag_with_varying_alpha(example_plain_env, cesaroni_m1670):
    """Test that 3D drag responds to angle of attack changes."""
    # Create drag function with strong alpha dependency
    mach = np.array([0.0, 0.5, 1.0, 1.5])
    reynolds = np.array([1e5, 1e6])
    alpha = np.array([0.0, 5.0, 10.0, 15.0])

    M, _, A = np.meshgrid(mach, reynolds, alpha, indexing="ij")
    # Strong alpha dependency: Cd increases significantly with alpha
    cd_data = 0.3 + 0.05 * M + 0.03 * A
    cd_data = np.clip(cd_data, 0.2, 2.0)

    drag_func = Function.from_grid(
        cd_data,
        [mach, reynolds, alpha],
        inputs=["Mach", "Reynolds", "Alpha"],
        outputs="Cd",
    )

    # Test at different angles of attack (direct function call)
    # At zero alpha, Cd should be lower
    cd_0 = drag_func(0.8, 5e5, 0.0)
    cd_10 = drag_func(0.8, 5e5, 10.0)

    # Cd should increase with alpha
    assert cd_10 > cd_0
    assert cd_10 - cd_0 > 0.2  # Should show significant difference

    # --- Integration test: verify flight uses alpha-dependent Cd ---
    # Create a flat (alpha-agnostic) drag by averaging over alpha
    cd_flat = cd_data.mean(axis=2)
    drag_flat = Function.from_grid(
        cd_flat,
        [mach, reynolds],
        inputs=["Mach", "Reynolds"],
        outputs="Cd",
    )

    # Use fixtures for environment and motor
    env = example_plain_env
    env.set_atmospheric_model(type="standard_atmosphere")
    motor = cesaroni_m1670

    # Build two rockets: one that uses alpha-sensitive drag, one flat
    rocket_alpha = Rocket(
        radius=0.0635,
        mass=16.24,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=drag_func,
        power_on_drag=drag_func,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket_alpha.set_rail_buttons(0.2, -0.5, 30)
    rocket_alpha.add_motor(motor, position=-1.255)

    rocket_flat = Rocket(
        radius=0.0635,
        mass=16.24,
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=drag_flat,
        power_on_drag=drag_flat,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket_flat.set_rail_buttons(0.2, -0.5, 30)
    rocket_flat.add_motor(motor, position=-1.255)

    # Run flights
    flight_alpha = Flight(
        rocket=rocket_alpha,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    flight_flat = Flight(
        rocket=rocket_flat,
        environment=env,
        rail_length=5.2,
        inclination=85,
        heading=0,
    )

    # Flights should both launch
    assert flight_alpha.apogee > 100
    assert flight_flat.apogee > 100

    # The two rockets should behave differently since one depends on alpha
    # while the other uses a flat-averaged Cd. Do not assume which direction
    # is larger (depends on encountered alpha vs averaged alpha) but ensure
    # the apogees differ.
    assert flight_alpha.apogee != flight_flat.apogee

    # Additionally, sample Cd during flight from aerodynamic state and
    # compare values computed from each rocket's drag function at the
    # same time index to ensure alpha actually affects the evaluated Cd.
    # Use mid-ascent index (avoid t=0). Find a time index where speed > 5 m/s
    speeds = flight_alpha.free_stream_speed[:, 1]
    times = flight_alpha.time
    idx_candidates = np.where(speeds > 5)[0]
    assert idx_candidates.size > 0
    idx = idx_candidates[len(idx_candidates) // 2]
    t_sample = times[idx]

    mach_sample = flight_alpha.mach_number.get_value_opt(t_sample)
    rho_sample = flight_alpha.density.get_value_opt(t_sample)
    mu_sample = flight_alpha.dynamic_viscosity.get_value_opt(t_sample)
    V_sample = flight_alpha.free_stream_speed.get_value_opt(t_sample)
    reynolds_sample = rho_sample * V_sample * (2 * rocket_alpha.radius) / mu_sample
    alpha_sample = flight_alpha.angle_of_attack.get_value_opt(t_sample)

    cd_alpha_sample = rocket_alpha.power_on_drag.get_value_opt(
        mach_sample, reynolds_sample, alpha_sample
    )
    cd_flat_sample = rocket_flat.power_on_drag.get_value_opt(
        mach_sample, reynolds_sample
    )

    # Alpha-sensitive Cd should differ from the flat Cd at the sampled time
    assert cd_alpha_sample != cd_flat_sample

    # Ensure the sign of the Cd difference is consistent with the apogee
    # ordering: larger Cd -> lower apogee
    if cd_alpha_sample > cd_flat_sample:
        assert flight_alpha.apogee < flight_flat.apogee
    else:
        assert flight_alpha.apogee > flight_flat.apogee
