from setuptools import setup
setup(name='OpenAttHetRL',
version='0.1',
description='attributed heterogeneous networks representation learning',
url='https://github.com/etemadir/OpenAttHetRL',
author='Roohollah Etemadi et al.',
author_email='etemadir@ryerson.ca',
license='MIT',
packages=['OpenAttHetRL'],
install_requires=['numpy==1.18.5', 'tensorflow', 'networkx==2.4','pickle5','simplejson','scikit-learn','matplotlib','beautifulsoup4','lxml','nltk==3.5','regex'],
zip_safe=False) 