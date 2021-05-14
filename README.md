# Powerfolio!

<img src="images/logo.png" width="150" title="stock_flex">

---
Creators: 
- Illya Nayshevsky, Ph.D.
- Nathan S. Froemming, Ph.D.
- Ludovic Schneider
- Chandra Kandiah


## Overview

---


## Theory
### Technical Analysis
#### RSI
Relative Strenght Index (RSI)

The relative strength index (RSI) is a momentum technical indicator used to evaluate overbought or oversold conditions in the price of securities. It measures the magnitude of price change of the asset. The range for RSI indicator is 0 to 100. [Ref. Investopedia.com](https://www.investopedia.com/terms/r/rsi.asp)


#### MACD



## Fundamental Analysis
#### PE

#### EPS

#### Dividnent



### Traditional Analysis
#### 

---

## Dependencies
### Anaconda
* [Anaconda Overview](https://docs.anaconda.com/)

### pandas
pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. [[Pandas Overview](https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html)]

#### Installation
```python
conda install pandas
```



### NumPy
The fundamental package for scientific computing with Python [[Numpy Overview](https://numpy.org/)]

#### Installation
```python
conda install numpy
```



### Panel
A high-level app and dashboarding solution for Python [[Holoviz Panel Overview](https://panel.holoviz.org/)]

#### Installation
```python
conda install -c pyviz panel
```



### SQLite
SQLite is a self-contained, high-reliability, embedded, full-featured, public-domain, SQL database engine. [[SQLite Overview](https://anaconda.org/anaconda/sqlite)]

#### Installation
```python
conda install -c anaconda sqlite
```



### SQLAlchemy
SQLAlchemy is the Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL. [[SQLAlchemy Overview](https://www.sqlalchemy.org/)]

#### Installation
```python
conda install -c anaconda sqlalchemy
```


### Plotly
Interactive charts and maps for Python, R, and JavaScript. [[Plotly Overview](https://plotly.com/python/getting-started/)]

#### Installation
Plotly must be installed inside the environment where it is meant to be used.

```python
conda install -c plotly plotly
```

<code>ipywidgets</code> and Jupyter Lab extentions must be added prior to Plotly's use

```python
conda install "notebook>=5.3" "ipywidgets>=7.5"
jupyter labextension install jupyterlab-plotly@4.14.3
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.14.3
```



### Matplotlib
Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. [[Matplotlib Overview](https://matplotlib.org/)]

#### Installation
```python
conda install matplotlib
```

### Alpaca Trade API
Alpaca is a technology company headquartered in Silicon Valley that builds commission-free stock trading API (Brokerage services are provided by Alpaca Securities LLC, member FINRA/SIPC) [[Alpaca Trade API Overview](https://alpaca.markets/docs/)]

#### Installation
```python
pip install alpaca-trade-api
```


### Quandl
The premier source for financial, economic and alternative datasets, serving investment professionals. Nasdaq’s Quandl platform is used by analysts from the world’s top hedge funds, asset managers and investment banks. [[Quandl API Overview](https://www.quandl.com/docs-and-help)]

#### Installation
```python
conda install -c anaconda quandl
```



### FinnHub
Real-Time RESTful APIs and Websocket for Stocks, Currencies, and Crypto. [[FinnHub API Overview](https://finnhub.io/docs/api/introduction)]

#### Installation
```python
pip install finnhub-python
```



## Computation Methods
