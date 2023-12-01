from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="vutils",
    version="1.0.7.dev123",
    author="Vichayturen",
    author_email="vichayturen@gmail.com",
    description="vichayturen's utils",
    url="https://github.com/RankKCodeTalker/myutils", 
    license="MIT",
    install_requires=[],
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=cythonize("src/vutils/**/*.py")
)