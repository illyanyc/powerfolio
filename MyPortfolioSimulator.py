from pandas import DataFrame
from typing import List


def get_tickers(df_alpaca: DataFrame, column_level: int = 0) -> List[str]:
    """
    Helper function to get the ticker symbols contained in an input dataframe
    originally created using the Alpaca API (i.e. the ticker symbols are the
    0th level of the dataframe's column `MultiIndex`).
    """
    return df_alpaca.columns.get_level_values(column_level).unique().tolist()


def get_attributes(df_alpaca: DataFrame, column_level: int = 1) -> List[str]:
    """
    Helper function to get the attributes contained in an input dataframe
    originally created using the Alpaca API, for example, ['open', 'high',
    'low', 'close', 'volume', 'macd', 'rsi', 'sharpe_ratio', ...].
    """
    return df_alpaca.columns.get_level_values(column_level).unique().tolist()


class MyPortfolioSimulator:
    """
    """

    def __init__(self):
        pass


def test():
    pass


def main():
    pass


if __name__ == 'main':
    main()
