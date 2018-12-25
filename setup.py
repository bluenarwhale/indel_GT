from distutils.core import setup
from Cython.Build import cythonize

setup(
        imixt_modules = cythonize('ModelPostProcess.pyx'),
)
