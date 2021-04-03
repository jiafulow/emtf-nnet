import setuptools

install_requires = [
    'numpy',
    'six',
]

extras_require = {
    'code-style': [
        'flake8',
        'flake8-bugbear',
        #'isort',
        #'black',
        'pre-commit',
    ],
    'tests': [
        'pytest',
    ],
    'docs': [],
}

extras_require['complete'] = sorted({v for req in extras_require.values() for v in req})

setuptools.setup(
    install_requires=install_requires,
    extras_require=extras_require,
)
