from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

_PACKAGE_NAME = 'onnx2torch'
_DIR_PATH = Path(__file__).parent.resolve()


def _get_installation_requirements():
    with _DIR_PATH.joinpath('requirements.txt').open('r', encoding='utf-8') as file:
        return list(line.strip() for line in file.readlines() if line and not line.startswith('#'))


def _get_version() -> str:
    with _DIR_PATH.joinpath(_PACKAGE_NAME, 'VERSION').open('r', encoding='utf-8') as version_file:
        version = version_file.read().strip()

    return version


def _get_long_description() -> str:
    with open('README.md', encoding='utf-8') as file:
        long_description = file.read()
    return long_description


setup(
    name=_PACKAGE_NAME,
    version=_get_version(),
    author='ENOT LLC',
    author_email='enot@enot.ai',
    license="apache-2.0",
    keywords=['AI', 'onnx', 'torch', 'onnx2torch', 'converters'],
    description='Nice Onnx to Pytorch converter',
    long_description=_get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/ENOT-AutoDL/onnx2torch',
    install_requires=_get_installation_requirements(),
    packages=find_packages(where=_DIR_PATH.as_posix()),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
