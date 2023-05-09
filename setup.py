import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RNA-LifeTime",
    version="1.0.0",
    author="Hua Zhenge",
    author_email="zheng.hua1@northeastern.edu",
    description="vLab is physics based deep learning model for RNA molecule lifetime prediction task",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='LICENSE',
    # url="https://github.com/pypa/sampleproject",
    project_urls={
        # "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=[],
    packages=setuptools.find_packages(where="RNALifeTime"),
    python_requires=">=3.7",
    test_suite='src.tests',
    package_data={'': ['MD-simulation/contact_matrix.pkl',
                       'MD-simulation/contacts.pkl',
                       'MD-simulation/location.pkl',
                       'MD-simulation/Mg2_data.pkl',
                       'MD-simulation/residues.pkl',
                       'MD-simulation/seq_code.pkl',
                       'MD-simulation/temperature_data.pkl']},
    include_package_data=True,
    install_requires=[
        'torch',
        'numpy==1.21.6',
        'pandas'
        'tensorboard',
        'bs4',  # for processing MD simulation trajectory
        'keras',  # for processing MD simulation trajectory
        'mdtraj',  # for processing MD simulation trajectory
        'scikit-learn'  # for processing MD simulation trajectory
    ],
)