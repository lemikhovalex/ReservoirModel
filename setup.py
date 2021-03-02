from setuptools import setup
import pkg_resources

setup(
    name='petro_res_pack',
    url='https://github.com/lemikhovalex/ReservoirModel',
    version='0.0.1',
    author='Aleksandr Lemikhov',
    author_email='lemikhovalex@gmail.com',
    description='Package with gym-like env for petroleum reservoir simulation',
    packages=['petro_res_pack'],
    license='MIT',
    install_reqs=pkg_resources.parse_requirements('requirements.txt')
)
