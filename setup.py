from setuptools import setup

setup(
    name="niftivis",
    version="2021.04.13",
    py_modules=["niftivis"],
    install_requires=[
        "Click",
        "Pillow",
        "nibabel>=2.0",
        "numpy"
    ],
    entry_points="""
        [console_scripts]
        niftivis=niftivis:cli
    """,
)
