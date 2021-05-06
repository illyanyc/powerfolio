# Stock Flex

<img src="images/logo.png" width="150" title="stock_flex">

---


## Overview


## Theory
### Technical Analysis
#### RSI
Relative Strenght Index (RSI)

The relative strength index (RSI) is a momentum technical indicator used to evaluate overbought or oversold conditions in the price of securities. It measures the magnitude of price change of the asset. The range for RSI indicator is 0 to 100.[Ref. Investopedia.com](https://www.investopedia.com/terms/r/rsi.asp)


## Installation
### Installing Plotly to run on Jupyter
* [Getting Started](https://plotly.com/python/getting-started/)

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

### Installing BeautifulSoup4 in Anaconda
* [Installing lxml](https://lxml.de/installation.html)
Installing lxml as a pre-requisite is necessary for parsing html tables into pythonic code
```python
pip install lxml
```

* [Installing bs4](https://anaconda.org/anaconda/beautifulsoup4)
```python
conda install -c anaconda beautifulsoup4
```


## Computation Methods