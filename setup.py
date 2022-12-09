import setuptools
import glob
import os


def process_requirements(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    with open(path, "r", encoding="utf-8") as f:
        requirements = f.read()
    processed_requirements = [x for x in requirements.strip().split("\n")]
    return processed_requirements


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="rutransform",
    version="0.0.1",
    author="evtaktasheva",
    author_email="evtaktasheva@gmail.com",
    description="Adversarial text perturbation framework for Russian",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RussianNLP/rutransform",
    packages=setuptools.find_packages(),
    package_data={"": ["*.json"]},
    include_package_data=True,
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=process_requirements("requirements.txt"),
)
