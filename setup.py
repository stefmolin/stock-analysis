from distutils.core import setup

setup(
    name='stock_analysis',
    version='0.0',
    description='Classes for technical analysis of stocks.',
    author='Stefanie Molin',
    author_email='24376333+stefmolin@users.noreply.github.com',
    license='MIT',
    url='https://github.com/stefmolin/stock-analysis',
    packages=['stock_analysis'],
    install_requires=[
          'pandas>=0.23.4',
          'pandas-datareader==0.7.0'
    ],
)
