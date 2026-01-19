from setuptools import setup

setup(
    name='abcd_rf_fit',
    version='0.1.2',
    description='Rational function fit library',
    url='https://github.com/UlysseREGLADE/abcd_rf_fit',
    author='Ulysse REGLADE',
    author_email='ulysse.reglade@yahoo.fr',
    license='License :: OSI Approved :: BSD License',
    packages=['abcd_rf_fit'],
    install_requires=['numpy',
                      'matplotlib',                     
                      'scipy'],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
],
)
