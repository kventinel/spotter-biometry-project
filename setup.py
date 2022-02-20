from pathlib import Path
from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def main():
    package_name = 'spotter'
    root_dir = Path(__file__).parent

    with open(root_dir / 'README.md') as f:
        long_description = f.read()

    with open(root_dir / 'requirements.txt') as f:
        reqs = [str(req) for req in parse_requirements(f)]

    packages = find_packages(package_name)
    packages = list(map(lambda x: f'{package_name}/{x}', packages))

    setup(
        name=package_name,
        version='0.0.1',
        author='sergevkim',
        description=package_name,
        long_description=long_description,
        long_description_content_type='text/markdown',
        package_dir={package_name: package_name},
        packages=packages,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.7',
        install_requires=reqs,
    )


if __name__ == '__main__':
    main()
