import setuptools

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

with open("requirements.txt", "r", encoding="utf-8") as require:
    requirements = [r for r in require.read().splitlines() if r != '']

setuptools.setup(
    name="sgl-dair",
    version="0.1.5",
    author="DAIR Lab @PKU",
    description="Graph Neural Network (GNN) toolkit targeting scalable graph learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PKU-DAIR/SGL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=requirements,
    data_files=["requirements.txt"],
)
