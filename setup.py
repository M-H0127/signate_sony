import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="signate_sony",
    version="0.1.0",
    author="higashi-masaki",
    author_email="ls16287j@gmail.com",
    description="predict pm2.5 with LGBM",
    url="https://github.com/M-H0127/signate_sony",
    packages=setuptools.find_packages(),
    #install_requires=["numpy", "scikit-learn", "torch", "transformers", "tqdm", "unidic-lite", "unidic", "fugashi", "ipadic"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)