from setuptools import setup

setup(
    name='PetroResPack',
    version='0.0.1',
    author='Aleksandr Lemikhov',
    author_email='lemikhovalex@gmail.com',
    description='Package with gym-like env for petroleum reservoir simulation',
    py_modules=['petro_env', 'session'],
    package_dir={'': 'src'}
)
