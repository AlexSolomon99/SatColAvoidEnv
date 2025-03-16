from setuptools import setup

setup(
    name="gym_satellite_ca",
    version="0.1.0",
    install_requires=[
        "python==3.11.8",
        "gymnasium==0.29.1",
        "orekit==12.0.1",
        "numpy==1.26.4"
    ],
    packages=["gym_satellite_ca", "gym_satellite_ca.envs"],
    include_package_data=True
)
