from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="ai_toolkit",
    version="1.0.0",
    description="THE GOD-TIER OMNIPOTENT AI FORGE",
    author="ereezyy",
    packages=find_packages(),
    py_modules=["ai_toolkit"],
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "ai-toolkit=ai_toolkit:cli",
        ],
    },
)
