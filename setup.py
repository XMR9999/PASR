import shutil

from setuptools import setup, find_packages

import pasr

PASR_VERSION = pasr.__version__

# 复制配置文件到项目目录下
shutil.rmtree('./pasr/configs/', ignore_errors=True)
shutil.copytree('./configs/', './pasr/configs/')


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_requirements(fname):
    with open(fname, encoding="utf-8-sig") as f:
        requirements = f.readlines()
    return requirements


if __name__ == "__main__":
    setup(
        name='pasr',
        packages=find_packages(exclude='download_data/'),
        package_data={'': ['configs/*']},
        author='yeyupiaoling',
        version=PASR_VERSION,
        install_requires=parse_requirements('./requirements.txt'),
        description='Automatic speech recognition toolkit on Pytorch',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='https://github.com/yeyupiaoling/PASR',
        download_url='https://github.com/yeyupiaoling/PASR.git',
        keywords=['asr', 'pytorch'],
        classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Natural Language :: Chinese (Simplified)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Utilities'
        ],
        license='Apache License 2.0',
        ext_modules=[])
    shutil.rmtree('./pasr/configs/', ignore_errors=True)
