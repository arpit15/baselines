from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


install_requires=[
          'gym[mujoco,atari,classic_control]',
          'scipy',
          'tqdm',
          'joblib',
          'zmq',
          'dill',
          # 'tensorflow == 1.0.0',
          'azure==1.0.3',
          'progressbar2',
          'mpi4py',
      ]


try:
    import tensorflow
except ImportError:
    install_requires.append('tensorflow')


setup(name='baselines',
      packages=[package for package in find_packages()
                if package.startswith('baselines')],
      install_requires=install_requires,
      description="OpenAI baselines: high quality implementations of reinforcement learning algorithms",
      author="OpenAI",
      url='https://github.com/openai/baselines',
      author_email="gym@openai.com",
      version="0.1.4")

