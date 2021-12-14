from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

_PACKAGE_NAME = 'onnx2torch'
_DIR_PATH = Path(__file__).parent.resolve()


def _get_installation_requirements():
    with _DIR_PATH.joinpath('requirements.txt').open('r') as file:
        return list(
            line.strip()
            for line in file.readlines()
            if line and not line.startswith('#')
        )


def _get_version() -> str:
    with _DIR_PATH.joinpath(_PACKAGE_NAME, 'VERSION').open('r') as version_file:
        version = version_file.read().strip()

    return version


setup(
    name=_PACKAGE_NAME,
    version=_get_version(),
    author='ENOT LLC',
    author_email='enot@enot.ai',
    install_requires=_get_installation_requirements(),
    packages=find_packages(where=_DIR_PATH.as_posix()),
    include_package_data=True,
)
