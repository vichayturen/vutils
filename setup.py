from setuptools import setup, find_packages

setup(
    name="myutils",
    version="1.0",
    author="Vichayturen",
    author_email="vichayturen@gmail.com",
    description="",

    # 项目主页
    url="", 
    # 表明当前模块依赖哪些包，若环境中没有，则会从pypi中下载安装
    install_requires=[
        'pandas'
    ],
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)