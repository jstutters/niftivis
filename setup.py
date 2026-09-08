from setuptools import setup

setup(
    name="niftivis",
    version="2021.04.13",
    python_requires=">=3.11",
    py_modules=["niftivis"],
    install_requires=[
        "Click>=8.3.3",
        "Pillow>=12.3.0",
        "nibabel>=5.3.2",
        "numpy>=1.22.0"
    ],
    entry_points="""
        [console_scripts]
        niftivis=niftivis:cli
    """,
)
