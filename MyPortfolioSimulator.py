import numpy as np
from pandas import DataFrame, MultiIndex
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


def get_log_returns(
    df_alpaca: DataFrame,
    input_label: str = 'close',
    output_label: str = 'log_return',
    diff_period: int = 1,
) -> DataFrame:
    """
    Helper function to calculate the logarithmic returns of an input dataframe.
    """
    # Check that the user's input label is present in the dataframe
    if not (input_label in get_attributes(df_alpaca)):
        raise ValueError(f"Input label \"{input_label}\" not found in "
                         f"dataframe column labels!")

    # Cache shorthands (for brevity & clarity)
    df = df_alpaca.loc[:, (slice(None), input_label)]
    tickers = get_tickers(df)

    # Calculate logarithmic returns
    df_log_returns = np.log(df / df.shift(periods=diff_period))
    df_log_returns.iloc[0, :] = 0.0  # replace leading `NaN` with correct value

    # Set pandas multiindex
    df_log_returns.columns = MultiIndex.from_product([tickers, [output_label]])

    # Return the result
    return df_log_returns


def get_sharpe_ratios(
    df_alpaca: DataFrame,
    input_label: str = 'close',
    output_label: str = 'sharpe_ratio',
    risk_free_rate: float = 0.0,
) -> DataFrame:
    """
    Helper function to calculate the Sharpe Ratio (SR) of the input portfolio
    dataframe.
    """
    # Calculate returns
    df_returns = get_log_returns(df_alpaca, input_label=input_label)

    # Calculate Sharpe Ratio
    df_sharpe_ratios = (df_returns.mean(axis=0) - risk_free_rate) / df_returns.std(axis=0)

    # Convert from `pandas.Series` to `pandas.DataFrame`, with appropriately
    # formatted columns
    tickers = get_tickers(df_alpaca)
    df_sharpe_ratios = df_sharpe_ratios.to_frame().T
    df_sharpe_ratios.columns = MultiIndex.from_product([tickers, [output_label]])

    # Return the results
    return df_sharpe_ratios


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
