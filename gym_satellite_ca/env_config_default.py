from gym_satellite_ca.envs.sat_data_class import SatelliteData


def get_sat_data_default_config():
    return {
        "sma": 6795000.0,
        "ecc": 0.00048,
        "inc": 51.6413,
        "argp": 21.0174,
        "raan": 60,
        "tran": 0.0,
        "mass": 100.0,
        "area": 1.0,
        "reflection_idx": 2.0,
        "angle_type": "DEGREE",
        "thruster_max_force": 0.01,
        "thruster_isp": 4000.0}


def get_env_default_kwargs():
    # the kwargs are: satellite: satellite: SatelliteData, ref_time: AbsoluteDate, ref_frame: Frame
    # set the satellite
    sat_data_config = get_sat_data_default_config()

    iss_satellite = SatelliteData(**sat_data_config)
    iss_satellite.change_angles_to_radians()

    default_kwargs = {"satellite": iss_satellite}

    return default_kwargs

