import numpy as np
from pandas import DataFrame, MultiIndex
from typing import List, NewType
from dataclasses import dataclass
from scipy.optimize import minimize


# Helpful user-defined type for portfolio-optimization simulations
Tickers = NewType('Tickers', List[str])
Weights = NewType('Weights', np.array)

@dataclass
class PortfolioOptimizationResult:
    tickers: Tickers = None
    weights: Weights = None
    periods_per_annum: int = None
    expected_return: float = None
    expected_variance: float = None
    descrip: str = None

    def __str__(self):
        descrip = 'PortfolioOptimizationResult' if self.descrip is None else self.descrip
        return (
            f"{descrip}\n"
            f"{'-' * len(descrip)}\n"
            f"tickers: {self.tickers}\n"
            f"weights: {self.weights}\n"
            f"periods_per_annum: {self.periods_per_annum}\n"
            f"expected_return: {self.expected_return}\n"
            f"expected_variance: {self.expected_variance}\n"
        )


def get_tickers(df_alpaca: DataFrame, column_level: int = 0) -> Tickers:
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
        allow_shorts: bool = False,
        input_label: str = 'close',
        debug: bool = False,
    ):
        # Assign member data
        self.keep_best_sharpe_ratios = keep_best_sharpe_ratios
        self.risk_free_rate = risk_free_rate
        self.periods_per_annum = periods_per_annum
        self.allow_shorts = allow_shorts
        self.input_label = input_label
        self.debug = debug

        # Get ticker symbols of financial instruments with the highest Sharpe Ratios
        self.df_sharpe_ratios = get_sharpe_ratios(df_alpaca, risk_free_rate=risk_free_rate, periods_per_annum=periods_per_annum).squeeze().sort_values(ascending=True).dropna()[-keep_best_sharpe_ratios:]
        self.tickers = get_tickers(self.df_sharpe_ratios.to_frame().T)

        # Calculate expected returns and covariance
        df_returns = get_log_returns(df_alpaca.loc[:, (self.tickers, self.input_label)])
        df_returns.columns = self.tickers
        self.df_returns_mean = df_returns.mean(axis=0)  # daily
        self.df_returns_cov = df_returns.cov()  # daily

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

    def get_equal_weight_portfolio(self) -> PortfolioOptimizationResult:
        # Calculate weights for equal-weight portfolio
        ntickers = len(self.tickers)
        weights = np.full(ntickers, (1.0 / ntickers))

        # Return portfolio information
        return PortfolioOptimizationResult(
            tickers = self.tickers,
            weights = weights,
            periods_per_annum = self.periods_per_annum,
            expected_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum),
            expected_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum),
            descrip = 'Equal-Weight Portfolio',
        )

    def get_minimum_variance_portfolio_analytical(self) -> PortfolioOptimizationResult:
        # Calculate the minimum-variance portfolio analytically
        ntickers = len(self.tickers)
        ones = np.ones(ntickers)
        covmat_inv = np.linalg.inv(self.df_returns_cov)
        weights = (covmat_inv @ ones) / (ones @ covmat_inv @ ones)
        if not self.allow_shorts:
            weights[np.where(weights < 0, True, False)] = 0.0
        weights /= np.sum(np.abs(weights))

        # Return the minimum-variance portfolio
        return PortfolioOptimizationResult(
            tickers = self.tickers,
            weights = weights,
            periods_per_annum = self.periods_per_annum,
            expected_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum),
            expected_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum),
            descrip = 'Minimum-Variance Portfolio (Analytic Solution -- Be Careful!)',
        )

    def get_minimum_variance_portfolio(self) -> PortfolioOptimizationResult:
        # Define arguments for `scipy.optimize.minimize`
        covmat = self.df_returns_cov.to_numpy()
        ftol = np.mean(covmat) / 1e5
        args = (covmat, self.periods_per_annum)

        # Perform the optimization
        opt_res = minimize(
            get_portfolio_variance,
            x0 = self._get_random_initial_weights(),
            bounds = self._get_weight_bounds_for_scipy(),
            constraints = self._get_weight_constraint_for_scipy(),
            args = args,
            method = 'SLSQP',
            options = {
                'ftol': ftol,
                'maxiter': 1e4,
            }
        )

        # Check that the optimization terminated successfully
        if not opt_res.success:
            raise RuntimeError(f"Optimization did not terminate successfully!")

        # Get optimal weights
        weights = opt_res.x
        if not np.isclose(np.sum(np.abs(weights)), 1.0):
            raise RuntimeError(f"sum(|weights|) do not total unity!")

        # Return the minimum-variance portfolio
        return PortfolioOptimizationResult(
            tickers = self.tickers,
            weights = weights,
            periods_per_annum = self.periods_per_annum,
            expected_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum),
            expected_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum),
            descrip = 'Minimum-Variance Portfolio',
        )

    def _get_random_initial_weights(self) -> Weights:
        ntickers = len(self.tickers)
        weights = np.random.random(ntickers)
        weights /= np.sum(np.abs(weights))
        if self.allow_shorts:
            weights *= np.random.choice([-1, 1], ntickers)
        return weights

    def _get_weight_bounds_for_scipy(self):
        return ((-1, 1) if self.allow_shorts else (0, 1), ) * len(self.tickers)

    def _get_weight_constraint_for_scipy(self):
        return {
            'type': 'eq',
            'fun': lambda weights: np.sum(np.abs(weights)) - 1
        }


def test():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
