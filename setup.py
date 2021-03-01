from setuptools import setup
from setuptools import find_packages

setup(
    name='petro_res_pack',
    url='https://github.com/lemikhovalex/ReservoirModel',
    version='0.0.1',
    author='Aleksandr Lemikhov',
    author_email='lemikhovalex@gmail.com',
    description='Package with gym-like env for petroleum reservoir simulation',
    py_modules=['petro_env', 'petro_session'],
    license='MIT',
)
