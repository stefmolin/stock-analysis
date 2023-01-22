from distutils.core import setup

setup(
    name='stock_analysis',
    version='0.2.1',
    description='Classes for technical analysis of stocks.',
    author='Stefanie Molin',
    author_email='24376333+stefmolin@users.noreply.github.com',
    license='MIT',
    url='https://github.com/stefmolin/stock-analysis',
    packages=['stock_analysis'],
    install_requires=[
        'matplotlib>=3.0.2',
        'mplfinance>=0.12.7a4',
        'numpy>=1.15.2',
        'pandas>=0.23.4',
        'pandas-datareader>=0.7.0',
        'seaborn>=0.11.0',
        'statsmodels>=0.11.1',
        'yfinance>=0.2.4'
    ],
)
