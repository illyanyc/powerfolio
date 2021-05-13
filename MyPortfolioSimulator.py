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
    periods_per_annum: int = 252,
) -> DataFrame:
    """
    Helper function to calculate the annualized Sharpe Ratios of the financial
    instruments contained in the input dataframe.
    """
    # Here's what we're calculating
    df_sharpe_ratios = None

    # Calculate logarithmic returns
    df_returns = get_log_returns(df_alpaca, input_label=input_label)

    # Calculate annualized expected return over the risk-free rate
    df_sharpe_numer = (df_returns.mean(axis=0) - risk_free_rate) * periods_per_annum

    # Calculate annualized expected standard deviation
    df_sharpe_denom = df_returns.std(axis=0) * np.sqrt(periods_per_annum)

    # Calculate annualized Sharpe Ratio
    df_sharpe_ratios = df_sharpe_numer / df_sharpe_denom

    # Convert `Series` to `DataFrame`, with appropriate format for this project
    df_sharpe_ratios = df_sharpe_ratios.to_frame().T

    # Construct `MultiIndex` with appropriate format for this project
    tickers = get_tickers(df_sharpe_ratios)
    df_sharpe_ratios.columns = MultiIndex.from_product([tickers, [output_label]])

    # Return the results
    return df_sharpe_ratios


def get_portfolio_return(
    weights: np.array = None,
    expected_returns: np.array = None,
    periods_per_annum: int = 252,
) -> float:
    """ Helper function to calculate the annualized portfolio return. """
    return ((1 + (weights @ expected_returns)) ** periods_per_annum) - 1


def get_portfolio_variance(
    weights: np.array = None,
    covariance_matrix: np.array = None,
    periods_per_annum: int = 252,
) -> float:
    """ Helper function to calculate the annualized portfolio variance. """
    return (weights @ covariance_matrix @ weights) * periods_per_annum


class MyPortfolioSimulator:
    """
    """

    def __init__(
        self,
        df_alpaca: DataFrame,
        keep_best_sharpe_ratios: int = 3,
        risk_free_rate: float = 0.0,
        periods_per_annum: int = 252,
        input_label: str = 'close',
        debug: bool = False,
    ):
        # Assign member data
        self.keep_best_sharpe_ratios = keep_best_sharpe_ratios
        self.risk_free_rate = risk_free_rate
        self.periods_per_annum = periods_per_annum
        self.input_label = input_label
        self.debug = debug

        # Get ticker symbols of financial instruments with the highest Sharpe Ratios
        self.df_sharpe_ratios = get_sharpe_ratios(df_alpaca, risk_free_rate=risk_free_rate, periods_per_annum=periods_per_annum).squeeze().sort_values(ascending=True).dropna()[-keep_best_sharpe_ratios:]
        self.tickers = get_tickers(self.df_sharpe_ratios.to_frame().T)

        # Calculate expected returns and covariance
        df_returns = get_log_returns(df_alpaca.loc[:, (self.tickers, self.input_label)])
        df_returns.columns = self.tickers
        self.df_returns_mean = df_returns.mean(axis=0)  # daily
        self.df_returns_cov = df_returns.cov()

        # Inform user
        if self.debug:
            self.print()

    def print(self) -> None:
        """ Helper function to print member data. """
        print('Tickers')
        print('-------')
        print(self.tickers)
        print()
        print('Annualized Sharpe Ratios')
        print('------------------------')
        print(self.df_sharpe_ratios)
        print()
        if 0:
            print('Expected Mean Return')
            print('--------------------')
            print(self.df_returns_mean)
            print()
            print('Expected Return Covariance')
            print('--------------------------')
            print(self.df_returns_cov)
            print()
        return None


def test():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
