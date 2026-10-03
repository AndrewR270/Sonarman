"""
Acoustic Intelligence (ACINT) Profile Library for Vessel Classes.
Contains baseline acoustic signatures for submarines, surface combatants,
and commercial shipping.
"""

# Base dictionary for scale & easy extension
VESSEL_ACOUSTIC_PROFILES = {
    # --------------------------------------------------------------------------
    # Submarines (SSN / SSBN / SSK)
    # Low fundamentals, low blade counts, low ocean acoustic footprint
    # --------------------------------------------------------------------------
    "ssn_los_angeles": {
        "engine_frequency": 15.0,
        "blade_frequencies": [30.0, 45.0],
        "shaft_turn_rate": 1.2,
        "ocean_noise_level": 0.08,  # Extremely quiet acoustic signature
    },
    "ssn_akula": {
        "engine_frequency": 18.0,
        "blade_frequencies": [36.0, 54.0],
        "shaft_turn_rate": 1.5,
        "ocean_noise_level": 0.10,
    },
    "ssbn_ohio": {
        "engine_frequency": 12.0,
        "blade_frequencies": [24.0, 36.0],
        "shaft_turn_rate": 0.8,
        "ocean_noise_level": 0.05,  # Ultra-quiet strategic deterrent
    },
    # --------------------------------------------------------------------------
    # Commercial & Civilian Vessels
    # Higher fundamentals, noisier blade harmonics, larger displacement noise
    # --------------------------------------------------------------------------
    "cargo": {
        "engine_frequency": 30.0,
        "blade_frequencies": [60.0, 90.0],
        "shaft_turn_rate": 2.5,
        "ocean_noise_level": 0.20,
    },
    "tanker": {
        "engine_frequency": 20.0,
        "blade_frequencies": [40.0, 60.0],
        "shaft_turn_rate": 1.5,
        "ocean_noise_level": 0.25,
    },
    "passenger": {
        "engine_frequency": 50.0,
        "blade_frequencies": [100.0, 150.0],
        "shaft_turn_rate": 4.0,
        "ocean_noise_level": 0.15,
    },
    "tug": {
        "engine_frequency": 40.0,
        "blade_frequencies": [80.0, 120.0],
        "shaft_turn_rate": 3.0,
        "ocean_noise_level": 0.20,
    },
}
