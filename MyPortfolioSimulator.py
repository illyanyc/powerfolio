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


def get_portfolio_return(
    weights: np.array = None,
    expected_returns: np.array = None,
    periods_per_annum: int = 252,
) -> float:
    """ Helper function to calculate the annualized portfolio return. """
    return ((1 + (weights @ expected_returns)) ** periods_per_annum) - 1


class MyPortfolioSimulator:
    """
    """

    def __init__(
        self,
        df_alpaca_ohclv: DataFrame,
        risk_free_rate: float = 0.0,
        keep_best_sharpe_ratios: int = 5,
        debug: bool = False,
    ):
        # Get ticker symbols of the best financial instruments as measured by
        # the Sharpe Ratio
        sharpe_ratios = get_sharpe_ratios(df_alpaca_ohclv, risk_free_rate=risk_free_rate).squeeze().sort_values(ascending=True).dropna()
        self.best_sharpe_ratios = sharpe_ratios[-keep_best_sharpe_ratios:]

        # Get tickers of the best Sharpe Ratios
        self.tickers = get_tickers(self.best_sharpe_ratios.to_frame().T)

        # Calculate returns of the best financial instruments
        df_returns = get_log_returns(df_alpaca_ohclv.loc[:, (self.tickers, 'close')])
        df_returns.columns = self.tickers

        # Cache expected return and covariance
        self.df_returns_avg = df_returns.mean()
        self.df_returns_cov = df_returns.cov()

        # Cache other variables
        self.debug = debug

        # Inform user
        if self.debug:
            print('Tickers')
            print('-------')
            print(self.tickers)
            print()
            print('Best Sharpe Ratios')
            print('------------------')
            print(self.best_sharpe_ratios)
            print()
            print('Return Mean')
            print('-----------')
            print(self.df_returns_avg)
            print()
            print('Return Covariance')
            print('-----------------')
            print(self.df_returns_cov)
            print()

    def do_simulation(self, num_simulations: int = 500):
        if self.debug:
            print(f"`{self.do_simulation.__name__}()`")
            print(self.tickers)
            print()

        # Cache variables
        self.num_simulations = num_simulations
        self.simulation_returns_vec = np.full(self.num_simulations, np.nan)
        self.simulation_variances_vec = np.full(self.num_simulations, np.nan)
        ntickers = len(self.tickers)

        # Loop over num_simulations
        for nsim in range(self.num_simulations):
            # Generate random weights
            weights = np.random.random(ntickers)

            # Make sure the weight vector is normalized
            weights /= np.sum(weights)

            # Calculate expected portfolio return
            #portfolio_return = np.dot(weights, self.df_returns_avg.to_numpy())
            portfolio_return = weights @ self.df_returns_avg

            # Calculate portfolio variance
            portfolio_variance = weights @ self.df_returns_cov @ weights

            # Add data to lists
            self.simulation_returns_vec[nsim] = portfolio_return
            self.simulation_variances_vec[nsim] = portfolio_variance

            # Inform user
            if (nsim == 0) or (((nsim + 1) % 100) == 0):
                print(f"Simulation: ({nsim + 1} / {self.num_simulations})")
                print(f"portfolio_return: {portfolio_return}")
                print(f"portfolio_variance: {portfolio_variance}")
                print()

    def get_efficient_frontier(self):
        pass


def test():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
