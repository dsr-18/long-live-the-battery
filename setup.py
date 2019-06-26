from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['numpy>=1.16',
 					 'pandas>=0.24',
					 'h5py>=2.9.0',
					 'scipy>=1.2.1',
                     'plotly>=3.10.0',
                     'google-cloud-storage>=1.16.1',
					 'tensorflow-gpu>=2.0.0b0']
					 # for GPU, replace line above with:  
					 # 'tensorflow-gpu>=2.0.0a']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)