from setuptools import setup
import pkg_resources

setup(
    name='petro_res_pack',
    url='https://github.com/lemikhovalex/ReservoirModel',
    version='0.7.3',
    author='Aleksandr Lemikhov',
    author_email='lemikhovalex@gmail.com',
    description='Package with gym-like env for petroleum reservoir simulation',
    packages=['petro_res_pack'],
    license='MIT',
    long_description='Package for simple modeling of reservoir in order to experiment with RL',
    install_reqs=['numpy', 'matplotlib', 'seaborn', 'ipython', 'pandas', 'scipy', 'gym', 'setuptools', 'pyglet']
)
