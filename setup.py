from setuptools import setup, find_packages

setup(
    name='pocketmon',
    version='1.0.0',
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
        'pocketmon=pocketmon.predict:main'
        ]
    },
    author='Marta García, Karim Hamed, Ivon Sánchez',
    description='Pockétmon: Predict protein binding pockets using 3D CNNs!',
    url='https://github.com/martagarnt/pocketmon'
)

if setup is not None:
    print("""⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡤⠶⠖⠒⠶⠤⣄⡀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡰⠞⢉⢀⠀⡀⡀⡀⠀⠀⠀⠈⠓⢤⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠎⠀⣠⡠⢋⣬⠊⢄⠑⢀⠀⠀⠄⢀⠀⡹⣄⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⣠⣿⠟⠓⢙⣋⡓⠑⠐⢌⠎⠞⠲⠀⢤⣿⣿⡆⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡏⡀⡟⠛⢁⣴⣶⣿⣶⣼⡖⠀⢁⣀⣦⣴⣿⣿⣿⣿⡀            Setup installed successfully! 
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣷⣯⣿⣷⣿⢏⡴⡢⣌⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃              ^^  Enjoy Pockétmon  ^^
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⡟⠛⠿⠿⣿⡘⣄⢀⣼⢠⣿⣿⣿⣿⡿⠿⠿⠿⠛⢻⠆           May you catch a lot of pockets!
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣻⡀⠀⠀⠘⠿⣦⣯⣴⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⣞⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⣷⡀⠀⠀⠀⠀⠐⠂⠁⠀⠀⠀⠀⠀⠀⠀⢀⡾⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⢷⠦⢄⣀⣀⣀⣀⣠⠴⠚⠉⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠐⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
""")
else:
    print(f"An error occurred during installation.")
