from setuptools import setup, find_packages
from setuptools.command.install import install

def print_install_banner():
    try:
        from pocketmon.predict import print_banner
        print_banner(mode="install")
    except Exception:
        print("\n Pockétmon installed! Run `pocketmon -h` to get started.\n")

class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        print_install_banner()

setup(
    name='pocketmon',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'biopython==1.83',
        'numpy==1.24.4',
        'requests==2.31.0',
        'scipy==1.13.0',
        'matplotlib==3.8.0',
        'scikit-learn==1.3.0',
        'torch==2.2.0'
    ],
    entry_points={
        'console_scripts': [
            'pocketmon=pocketmon.predict:main',
        ],
    },
    cmdclass={'install': CustomInstallCommand},
    author='Marta García, Karim Hamed, Ivon Sánchez',
    description='Pockétmon: CNN-based tool for binding pocket prediction',
    url='https://github.com/martagarnt/pocketmon',
)