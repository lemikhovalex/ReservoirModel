from setuptools import setup

setup(
    name='PetroResPack',
    url='https://github.com/lemikhovalex/ReservoirModel',
    version='0.0.1',
    author='Aleksandr Lemikhov',
    author_email='lemikhovalex@gmail.com',
    description='Package with gym-like env for petroleum reservoir simulation',
    py_modules=['petro_env', 'petro_session'],
    packages=['petro_res_pack'],
    license='MIT'
)
