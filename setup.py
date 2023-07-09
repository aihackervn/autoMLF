from setuptools import find_packages, setup

with open("core/autoML/README.md", "r") as f:
    long_description = f.read()
with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f if line.strip()]
setup(
    name="autoMLF",
    version="1.0.3",
    description="autoML for training and inference Deep Learning model",
    package_dir={"": "core"},
    packages=find_packages(where="./core"),
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aihackervn",
    author="TinVo",
    author_email="tinprocoder0908@gmail.com",
    keywords=["AI HACKER VNESE", "AI", "autoML", "Deep Learning", "Computer Vision",
              "Interface"],

    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'auto_mls = autoML.CLI.automltoolkit:start_terminal_loop',
        ],
    },
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9"
)
