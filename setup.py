import setuptools

with open('README.md', 'r') as description_file:
    long_description = description_file.read()

setuptools.setup(
    name='kattention',
    version='0.1.1',
    author='Sofiene ALOUINI',
    author_email='sofiene.alouini@gmail.com',
    description='Package implementing different attention mechanisms as tf.keras layers',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sofienealouini/kattention',
    python_requires='>=3.7',
    install_requires=['tensorflow>=2.2'],
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
