from setuptools import setup, find_packages

setup(
    name="vutils",
    version="1.0.7",
    author="Vichayturen",
    author_email="vichayturen@gmail.com",
    description="vichayturen's utils",

    url="https://github.com/RankKCodeTalker/myutils", 
    license="MIT",
    install_requires=[],
    package_dir={"": "src"},
    packages=find_packages("src"),
)