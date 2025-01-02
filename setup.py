from setuptools import setup, find_packages


with open("VERSION", "r", encoding="utf-8") as f:
    version = f.read().strip()


setup(
    name="vutils",
    version=version,
    author="Vichayturen",
    author_email="vichayturen@gmail.com",
    description="vichayturen's utils",
    url="https://github.com/RankKCodeTalker/vutils",
    license="MIT",
    install_requires=[],
    extra_requires=[
        'numpy'
        'pandas',
        'openpyxl',
        'matplotlib',
        'scipy',
        'scikit-learn',
        'torch',
        'tqdm',
        'gradio',
        'openai',
    ],
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        "console_scripts": [
            "vvcli=vutils.main:main",
        ],
    },
)
