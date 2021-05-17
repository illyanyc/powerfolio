"""
TODO: Documentation.
"""
import numpy as np
import enum
from pandas import read_csv, Series, DataFrame, MultiIndex, concat
from typing import List, NewType, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from scipy.optimize import minimize


# Helpful user-defined type for portfolio-optimization simulations
Tickers = NewType('Tickers', List[str])
Weights = NewType('Weights', np.array)


@dataclass
class PortfolioOptimizationResult:
    descrip: str = None
    tickers: Tickers = None
    weights: Weights = None
    expected_return: float = None
    expected_variance: float = None
    periods_per_annum: int = None
    sharpe_ratio: float = None

    def __str__(self):
        descrip = 'PortfolioOptimizationResult' if self.descrip is None else self.descrip
        return (
            f"{descrip}\n"
            f"{'-' * len(descrip)}\n"
            f"tickers: {self.tickers}\n"
            f"weights: {self.weights}\n"
            f"expected_return: {self.expected_return}\n"
            f"expected_variance: {self.expected_variance}\n"
            f"periods_per_annum: {self.periods_per_annum}\n"
            f"sharpe_ratio: {self.sharpe_ratio}\n"
        )


class InterestingPortfolios(enum.IntEnum):
    """ Helper class to keep results for interesting portfolios straight! """
    MINIMUM_VARIANCE = 0
    MAXIMUM_SHARPE = 1
    EQUAL_WEIGHT = 2
    NPORTFOLIOS = enum.auto()


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
    df_returns: DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_annum: int = 252
) -> Series:
    """
    Helper function to calculate the (annualized) Sharpe Ratios of the financial
    instruments contained in the input dataframe.
    """
    numer = (df_returns.mean(axis=0) - risk_free_rate) * periods_per_annum
    denom = np.sqrt(df_returns.var(axis=0) * periods_per_annum)
    return numer / denom


def get_portfolio_return(
    weights: np.array = None,
    expected_returns: np.array = None,
    periods_per_annum: int = 252,
) -> float:
    """ Helper function to calculate the (annualized) portfolio return. """
    return (weights @ expected_returns) * periods_per_annum


def get_portfolio_variance(
    weights: np.array = None,
    covariance_matrix: np.array = None,
    periods_per_annum: int = 252,
) -> float:
    """ Helper function to calculate the (annualized) portfolio variance. """
    return (weights @ covariance_matrix @ weights) * periods_per_annum


def get_portfolio_sharpe_ratio(
    weights: np.array = None,
    expected_returns: np.array = None,
    covariance_matrix: np.array = None,
    risk_free_rate: float = 0.0,
    periods_per_annum: int = 252,
) -> float:
    idx_sharpe_ratio = -1
    return get_portfolio_return_variance_sharpe(
        weights=weights,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=risk_free_rate,
        periods_per_annum=periods_per_annum,
    )[idx_sharpe_ratio]


def get_portfolio_return_variance_sharpe(
    weights: np.array = None,
    expected_returns: np.array = None,
    covariance_matrix: np.array = None,
    risk_free_rate: float = 0.0,
    periods_per_annum: int = 252,
) -> Tuple[float, float, float]:
    """ Wrapper method to calculate a portfolio's expected {return, variance,
    Sharpe Ratio} with one function call. """
    portfolio_return = get_portfolio_return(
        weights=weights, expected_returns=expected_returns, periods_per_annum=periods_per_annum)
    portfolio_variance = get_portfolio_variance(
        weights=weights, covariance_matrix=covariance_matrix, periods_per_annum=periods_per_annum)
    portfolio_sharpe_numer = (
        portfolio_return - risk_free_rate) * periods_per_annum
    portfolio_sharpe_denom = np.sqrt(portfolio_variance * periods_per_annum)
    portfolio_sharpe_ratio = portfolio_sharpe_numer / portfolio_sharpe_denom
    return (portfolio_return, portfolio_variance, portfolio_sharpe_ratio)


def get_neg_portfolio_sharpe_ratio_for_scipy(
    weights: np.array = None,
    expected_returns: np.array = None,
    covariance_matrix: np.array = None,
    risk_free_rate: float = 0.0,
    periods_per_annum: int = 252,
) -> float:
    """
    In order to maximize the Sharpe Ratio using `scipy.optimize.minimize`,
    we will *minimize the negative Sharpe Ratio*.
    """
    return -get_portfolio_sharpe_ratio(
        weights = weights,
        expected_returns = expected_returns,
        covariance_matrix = covariance_matrix,
        risk_free_rate = risk_free_rate,
        periods_per_annum = periods_per_annum,
    )


class TraditionalPortfolioAnalyzer:
    """
    """

    def __init__(
        self,
        df_returns: DataFrame,
        risk_free_rate: float = 0.0,
        periods_per_annum: int = 252,
        allow_shorts: bool = True,
        debug: bool = True,
    ):
        self.risk_free_rate = risk_free_rate
        self.periods_per_annum = periods_per_annum
        self.allow_shorts = allow_shorts
        self.debug = debug
        self.update_returns(df_returns)

    def update_returns(self, df_returns: DataFrame) -> None:
        self.tickers = get_tickers(df_returns)
        self.df_returns = df_returns.copy()
        self.df_returns_mean = self.df_returns.mean(axis=0)
        self.df_returns_cov = self.df_returns.cov()
        self.sharpe_ratios = get_sharpe_ratios(
            self.df_returns, risk_free_rate=self.risk_free_rate, periods_per_annum=self.periods_per_annum)
        return None

    def get_individual_stock_portfolios(self) -> DataFrame:
        ntickers = len(self.tickers)
        weights = np.ones(ntickers)
        expected_returns = np.full(ntickers, np.nan)
        expected_variances = np.full(ntickers, np.nan)
        expected_sharpe_ratios = np.full(ntickers, np.nan)
        for n in range(ntickers):
            pret, pvar, psharpe = get_portfolio_return_variance_sharpe(
                weights=np.array([1.0]),
                expected_returns=np.array([self.df_returns_mean.iloc[n]]),
                covariance_matrix=np.array([[self.df_returns_cov.iloc[n, n]]]),
                risk_free_rate=self.risk_free_rate,
                periods_per_annum=self.periods_per_annum
            )
            expected_returns[n] = pret
            expected_variances[n] = pvar
            expected_sharpe_ratios[n] = psharpe
        return self._build_dataframe_for_hvplot(
            portfolio_type='one_stock',
            portfolio_return=expected_returns,
            portfolio_variance=expected_variances,
            portfolio_sharpe=expected_sharpe_ratios,
            tickers=self.tickers,
            weights=([str(1.0)] * ntickers),
        )

    def get_equal_weight_portfolio(self):
        ntickers = len(self.tickers)
        weights = np.full(ntickers, (1.0 / ntickers))
        pret, pvar, psharpe = get_portfolio_return_variance_sharpe(
            weights=weights,
            expected_returns=self.df_returns_mean,
            covariance_matrix=self.df_returns_cov,
            risk_free_rate=self.risk_free_rate,
            periods_per_annum=self.periods_per_annum,
        )
        return self._build_dataframe_for_hvplot(
            portfolio_type='equal_weight',
            portfolio_return=[pret],
            portfolio_variance=[pvar],
            portfolio_sharpe=[psharpe],
            tickers=','.join(self.tickers),
            weights=','.join([str(round(wt, 5)) for wt in weights]),
        )

    def get_random_portfolio(self) -> DataFrame:
        weights = self.get_random_initial_weights()
        pret, pvar, psharpe = get_portfolio_return_variance_sharpe(
            weights=weights,
            expected_returns=self.df_returns_mean,
            covariance_matrix=self.df_returns_cov,
            risk_free_rate=self.risk_free_rate,
            periods_per_annum=self.periods_per_annum,
        )
        return self._build_dataframe_for_hvplot(
            portfolio_type='random',
            portfolio_return=[pret],
            portfolio_variance=[pvar],
            portfolio_sharpe=[psharpe],
            tickers=','.join(self.tickers),
            weights=','.join([str(round(wt, 5)) for wt in weights]),
        )

    def get_random_portfolios(self, nrandom: int = 1000) -> DataFrame:
        dfs = []
        for _ in range(nrandom):
            dfs.append(self.get_random_portfolio())
        return concat(dfs)

    def do_portfolio_constrained_optimization(
        self,
        portfolio_type: str = 'minimum_variance',
        target_return: float = None,
    ):
        allowed_portfolio_types = {
            'minimum_variance',
            'maximum_sharpe',
            'mean_return_constraint'
        }
        if portfolio_type not in allowed_portfolio_types:
            raise ValueError(
                f"do_portfolio_constrained_optimization: `portfolio_type "
                f"= {portfolio_type}` not understood!  Allowed options: "
                f"= {allowed_portfolio_types}`"
            )

        if (portfolio_type == 'mean_return_constraint') and (target_return is None):
            raise ValueError(
                f"do_portfolio_constrained_optimization: `portfolio_type "
                f"= {portfolio_type}` must have !"
            )

        covmat = self.df_returns_cov.to_numpy()
        catol = 1.0e-5
        ftol = np.mean(covmat) * catol  # precision of optimization
        eps = ftol

        weights = self.get_random_initial_weights()
        weight_bounds = ((-1, 1) if self.allow_shorts else (0, 1), ) * len(self.tickers)

        if portfolio_type == 'minimum_variance':
            opt_fcn = get_portfolio_variance
            args = (covmat, self.periods_per_annum)
            constraints = self._get_weight_constraint_for_scipy()
        elif portfolio_type == 'maximum_sharpe':
            opt_fcn = get_neg_portfolio_sharpe_ratio_for_scipy
            args = (self.df_returns_mean.to_numpy(), covmat, self.risk_free_rate, self.periods_per_annum)
            constraints = self._get_weight_constraint_for_scipy()
        elif portfolio_type == 'mean_return_constraint':
            def _return_constraint_fcn(weights):  #, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum):
                return get_portfolio_return(
                    weights=weights,
                    expected_returns=self.df_returns_mean,
                    periods_per_annum=self.periods_per_annum
                )
            opt_fcn = get_portfolio_variance
            args = (covmat, self.periods_per_annum)
            constraints = [
                self._get_weight_constraint_for_scipy(),
                {'type': 'eq', 'fun': lambda wt: _return_constraint_fcn(wt) - target_return}
            ]
        else:
            print(f"")

        # Perform the optimization
        opt_res = minimize(
            opt_fcn,
            x0=weights,
            bounds=weight_bounds,
            constraints=constraints,
            args=args,
            method='SLSQP',
            options={
                'catol': catol,
                'ftol': ftol,
                'eps': eps,
                'maxiter': 1e4,
            }
        )

        # Check that the optimization terminated successfully
        if not opt_res.success:
            raise RuntimeError(
                f"get_minimum_variance_portfolio: "
                f"Optimization did not terminate successfully!"
            )

        # Get optimal weights
        weights = opt_res.x
        if not np.isclose(np.sum(np.abs(weights)), 1.0):
            raise RuntimeError(
                f"get_minimum_variance_portfolio: "
                f"sum(|weights|) do not total unity!"
            )

        pret, pvar, psharpe = get_portfolio_return_variance_sharpe(
            weights=weights,
            expected_returns=self.df_returns_mean,
            covariance_matrix=self.df_returns_cov,
            risk_free_rate=self.risk_free_rate,
            periods_per_annum=self.periods_per_annum,
        )

        return self._build_dataframe_for_hvplot(
            portfolio_type=portfolio_type,
            portfolio_return=[pret],
            portfolio_variance=[pvar],
            portfolio_sharpe=[psharpe],
            tickers=','.join(self.tickers),
            weights=','.join([str(round(wt, 5)) for wt in weights]),
        )

        return opt_res

    def get_minimum_variance_portfolio(self):
        return self.do_portfolio_constrained_optimization(portfolio_type='minimum_variance')

    def get_maximum_sharpe_portfolio(self):
        return self.do_portfolio_constrained_optimization(portfolio_type='maximum_sharpe')

    def get_efficient_frontier_portfolios(self, num_portfolios: int = 11):
        min_return = self.df_returns_mean.min()
        max_return = self.df_returns_mean.max()

        target_returns = np.linspace(min_return, max_return, num=(num_portfolios + 2))[1:-1] * self.periods_per_annum
        dfs = []
        for target_return in target_returns:
            dfs.append(self.do_portfolio_constrained_optimization(
                portfolio_type = 'mean_return_constraint',
                target_return=target_return
            ))
        df = concat(dfs, axis=0, join='outer')
        df['portfolio_type'] = 'efficient_frontier'
        return df

    def get_random_initial_weights(self) -> np.array:
        ntickers = len(self.tickers)
        weights = np.random.random(ntickers)
        if self.allow_shorts:
            weights *= np.random.choice((-1, 1), ntickers)
        weights /= np.sum(np.abs(weights))
        return weights

    def _build_dataframe_for_hvplot(
        self,
        portfolio_type: List[str] = None,
        portfolio_return: np.array = None,
        portfolio_variance: np.array = None,
        portfolio_sharpe: np.array = None,
        tickers: Tickers = None,
        weights: Weights = None,
    ):
        return DataFrame(
            data={
                'portfolio_type': portfolio_type,
                'portfolio_return': portfolio_return,
                'portfolio_variance': portfolio_variance,
                'portfolio_sharpe': portfolio_sharpe,
                'tickers': tickers,
                'weights': weights,
            },
        )

    def _get_weight_constraint_for_scipy(self):
        return {
            'type': 'eq',
            'fun': lambda weights: np.sum(np.abs(weights)) - 1
        }


# ===============================================================================
# HERE I AM!!! -----------------------------------------------------------------
# ===============================================================================
'''


class RememberToSleepDumbass:
    """
    """

        # Pick a few tickers if None is provided
        if tickers_of_interest is None:
            nbest_sharpe = 16
            self.set_tickers_of_interest(self.df_sharpe_ratios.droplevel(1, axis=1).squeeze().sort_values(ascending=True).dropna()[-nbest_sharpe:].index.tolist())
        
        # Get outta here
        return None

    def load_ohlcv_data(self, ohlcv_data_path: Path) -> None:
        """ Helper function to load raw {Open, High, Low, Close, Volume} data. """
        # Load raw data
        self.ohlcv_data_path = ohlcv_data_path
        df = read_csv(ohlcv_data_path, header=[0, 1], index_col=0, parse_dates=True, infer_datetime_format=True)

        # Calculate and cache other data (for fast dashboard)
        self.df_returns = get_log_returns(df)
        self.df_returns_mean = self.df_returns.mean(axis=0)
        self.df_returns_corr = self.df_returns.droplevel(1, axis=1).corr()
        self.df_returns_cov = self.df_returns.droplevel(1, axis=1).cov()
        self.df_sharpe_ratios = get_sharpe_ratios(df)
        return None

    def set_tickers_of_interest(self, tickers_of_interest: List[str]) -> None:
        self.tickers_of_interest = tickers_of_interest
        return None

    def get_tickers_of_interest(self) -> List[str]:
        return self.tickers_of_interest

    def get_returns_vector(self):
        tickers = self.tickers_of_interest
        return self.df_returns_mean[tickers].to_numpy()

    def get_covariance_matrix(self):
        tickers = self.tickers_of_interest
        return self.df_returns_cov.loc[tickers, tickers].to_numpy()

    def get_correlation_matrix(self):
        tickers = self.tickers_of_interest
        return self.df_returns_corr.loc[tickers, tickers]

    def get_distance_matrix(self):
        return np.sqrt(0.5 * (1.0 - self.get_correlation_matrix()))
    
    
        

        expected_return = self.df_returns_mean[tickers].to_numpy() * self.periods_per_annum

        covmat = self.df_returns_cov.loc[tickers, tickers]
        print(covmat.head())

        expected_variance = np.array([self.df_returns_cov.iloc[i, i] for i in range(len(tickers))]) * self.periods_per_annum

        df = DataFrame(
            data={
                'ticker': tickers,
                'expected_return': expected_return,
                'expected_variance': expected_variance,
                'sharpe_ratio': self.df_sharpe_ratios.squeeze()[tickers].tolist(),
            },
        )
        return df
    
    def run_traditional_portfolio_analysis(self, num_simulations: int = 4):
        """
        """
        # Here's what we're calculating
        self.sim_results = []

        # Run simulations
        for nsim in range(num_simulations):
            # Cache quantities used below
            weights = self._get_random_initial_weights()
            returns_vector = self.get_returns_vector()
            covariance_matrix = self.get_covariance_matrix()
            portfolio_return = get_portfolio_return(weights=weights, expected_returns=returns_vector, periods_per_annum=self.periods_per_annum)
            portfolio_variance = get_portfolio_variance(weights=weights, covariance_matrix=covariance_matrix, periods_per_annum=self.periods_per_annum)
            portfolio_sharpe_ratio = get_portfolio_sharpe_ratio(
                weights=weights,
                expected_returns=returns_vector,
                covariance_matrix=covariance_matrix,
                risk_free_rate=self.risk_free_rate,
                periods_per_annum=self.periods_per_annum,
            )
            # Append results to list
            self.sim_results.append(
                PortfolioOptimizationResult(
                    tickers = self.tickers_of_interest,
                    weights = weights,
                    expected_return = portfolio_return,
                    expected_variance = portfolio_variance,
                    periods_per_annum = self.periods_per_annum,
                    sharpe_ratio = portfolio_sharpe_ratio,
                )
            )

        # Combine results into dataframe
        nsims = range(num_simulations)
        return DataFrame({
            'expected_return': [self.sim_results[n].expected_return for n in nsims],
            'expected_variance': [self.sim_results[n].expected_variance for n in nsims],
            'sharpe_ratio': [self.sim_results[n].sharpe_ratio for n in nsims]
        })
    
    def get_equal_weight_portfolio(self) -> PortfolioOptimizationResult:
        ntickers = len(self.tickers_of_interest)
        weights = np.full(ntickers, (1.0 / ntickers))

        returns_vector = self.get_returns_vector()
        covariance_matrix = self.get_covariance_matrix()
        
        portfolio_return = get_portfolio_return(weights=weights, expected_returns=returns_vector, periods_per_annum=self.periods_per_annum)
        portfolio_variance = get_portfolio_variance(weights=weights, covariance_matrix=covariance_matrix, periods_per_annum=self.periods_per_annum)
        portfolio_sharpe_ratio = get_portfolio_sharpe_ratio(
            weights = weights,
            expected_returns = returns_vector,
            covariance_matrix = covariance_matrix,
            risk_free_rate = self.risk_free_rate,
            periods_per_annum = self.periods_per_annum,
        )
        return PortfolioOptimizationResult(
            tickers = self.tickers_of_interest,
            weights = weights,
            expected_return = portfolio_return,
            expected_variance = portfolio_variance,
            periods_per_annum = self.periods_per_annum,
            sharpe_ratio = portfolio_sharpe_ratio,
        )
    
    def get_minimum_variance_portfolio(self) -> PortfolioOptimizationResult:
        # Cache quantities and define shorthands
        covariance_matrix = self.get_covariance_matrix()
        ftol = np.mean(covariance_matrix) / 1e5  # precision of optimization
        args = (covariance_matrix, ftol)

        # Perform optimization
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
        
        # Now that we have the weights, calculate expected return, variance, etc.
        returns_vector = self.get_returns_vector()
        portfolio_return = get_portfolio_return(weights=weights, expected_returns=returns_vector, periods_per_annum=self.periods_per_annum)
        portfolio_variance = get_portfolio_variance(weights=weights, covariance_matrix=covariance_matrix, periods_per_annum=self.periods_per_annum)
        portfolio_sharpe_ratio = get_portfolio_sharpe_ratio(
            weights=weights,
            expected_returns=returns_vector,
            covariance_matrix=covariance_matrix,
            risk_free_rate=self.risk_free_rate,
            periods_per_annum=self.periods_per_annum,
        )
        # Return the result
        return PortfolioOptimizationResult(
            tickers = self.tickers_of_interest,
            weights = weights,
            expected_return = portfolio_return,
            expected_variance = portfolio_variance,
            periods_per_annum = self.periods_per_annum,
            sharpe_ratio = portfolio_sharpe_ratio,
        )

    def _get_random_initial_weights(self) -> Weights:
        ntickers = len(self.tickers_of_interest)
        weights = np.random.random(ntickers)
        weights /= np.sum(np.abs(weights))
        return weights
    
    def _get_weight_bounds_for_scipy(self):
        ntickers = len(self.tickers_of_interest)
        return ((-1, 1) if self.allow_shorts else (0, 1), ) * ntickers




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
        self.df_returns = get_log_returns(df_alpaca.loc[:, (self.tickers, self.input_label)])
        self.df_returns.columns = self.tickers
        self.df_returns_mean = self.df_returns.mean(axis=0)  # daily
        self.df_returns_cov = self.df_returns.cov()  # daily

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

        portfolio_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum)
        portfolio_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum)
        portfolio_sharpe_ratio = get_portfolio_sharpe_ratio(
            weights=weights,
            expected_returns=self.df_returns_mean,
            covariance_matrix=self.df_returns_cov,
            risk_free_rate=self.risk_free_rate,
            periods_per_annum=self.periods_per_annum,
        )

        # Return portfolio information
        return PortfolioOptimizationResult(
            descrip = 'Equal-Weight Portfolio',
            tickers = self.tickers,
            weights = weights,
            expected_return = portfolio_return,
            expected_variance = portfolio_variance,
            periods_per_annum = self.periods_per_annum,
            sharpe_ratio = portfolio_sharpe_ratio,
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
            descrip = 'Minimum-Variance Portfolio (Analytic Solution -- Be Careful!)',
            tickers = self.tickers,
            weights = weights,
            expected_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum),
            expected_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum),
            periods_per_annum = self.periods_per_annum,
            sharpe_ratio = 'TODO',
        )

    def get_minimum_variance_portfolio(self) -> PortfolioOptimizationResult:
        # Define arguments for `scipy.optimize.minimize`
        
        

        # Perform the optimization
        opt_res = minimize(
            get_portfolio_variance,
            x0 = self._get_random_initial_weights(),
            bounds = self._get_weight_bounds_for_scipy(),
            constraints = self._get_weight_constraint_for_scipy(),
            args = args,
            method = 'SLSQP',

        )

        # Return the minimum-variance portfolio
        return PortfolioOptimizationResult(
            descrip = 'Minimum-Variance Portfolio',
            tickers = self.tickers,
            weights = weights,
            expected_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum),
            expected_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum),
            periods_per_annum = self.periods_per_annum,
            sharpe_ratio = 'TODO',
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

    def get_maximum_sharpe_ratio_portfolio(self) -> PortfolioOptimizationResult:
        # Construct arguments for `get_neg_sharpe_ratio_for_scipy()`
        expected_returns = self.df_returns_mean.to_numpy()
        covariance_matrix = self.df_returns_cov.to_numpy()
        risk_free_rate = self.risk_free_rate
        periods_per_annum = self.periods_per_annum
        args = (
            expected_returns,
            covariance_matrix,
            risk_free_rate,
            periods_per_annum
        )

        ftol = np.mean(covariance_matrix) / 1e5

        opt_res = minimize(
            get_neg_portfolio_sharpe_ratio_for_scipy,
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
            descrip = 'Maximum-Sharpe-Ratio Portfolio',
            tickers = self.tickers,
            weights = weights,
            expected_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum),
            expected_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum),
            periods_per_annum = self.periods_per_annum,
            sharpe_ratio = 'TODO',
        )

    def get_mean_variance_bulk(self, num_simulations: int = 500) -> List[PortfolioOptimizationResult]:
        self.results = []
        for nsim in range(num_simulations):
            weights = self._get_random_initial_weights()
            portfolio_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum)
            portfolio_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum)
            portfolio_sharpe_ratio = get_portfolio_sharpe_ratio(
                weights=weights,
                expected_returns=self.df_returns_mean,
                covariance_matrix=self.df_returns_cov,
                risk_free_rate=self.risk_free_rate,
                periods_per_annum=self.periods_per_annum,
            )
            result = PortfolioOptimizationResult(
                tickers = self.tickers,
                weights = weights,
                expected_return = portfolio_return,
                expected_variance = portfolio_variance,
                periods_per_annum = self.periods_per_annum,
                sharpe_ratio = portfolio_sharpe_ratio,
            )
            self.results.append(result)
        return self.results

    def get_efficient_frontier(self, num_simulations: int = 51) -> List[PortfolioOptimizationResult]:



        # Determine constraints
        if self.allow_shorts:
            min_return = -self.df_returns_mean.abs().max() / 2
            max_return =  self.df_returns_mean.abs().max() / 2
        else:
            min_return = self.df_returns_mean.min() / 1
            max_return = self.df_returns_mean.max() / 2

        return_constraints = np.linspace(min_return, max_return, num=(num_simulations + 2))[1:-1]


        covmat = self.df_returns_cov.to_numpy()
        ftol = np.mean(covmat) / 1e4
        args = (covmat, self.periods_per_annum)

        efficient_frontier = []

        target_returns = return_constraints  * self.periods_per_annum

        for target_return in target_returns:

            print(f"target_return: {target_return}")

            opt_res = minimize(
                get_portfolio_variance,
                x0 = self._get_random_initial_weights(),
                bounds = self._get_weight_bounds_for_scipy(),
                constraints = [
                    self._get_weight_constraint_for_scipy(),
                    
                ],
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

            # Add this portfolio to the efficient frontier
            efficient_frontier.append(
                PortfolioOptimizationResult(
                    descrip = 'Constrained-Mean Portfolio',
                    tickers = self.tickers,
                    weights = weights,
                    expected_return = get_portfolio_return(weights=weights, expected_returns=self.df_returns_mean, periods_per_annum=self.periods_per_annum),
                    expected_variance = get_portfolio_variance(weights=weights, covariance_matrix=self.df_returns_cov, periods_per_annum=self.periods_per_annum),
                    periods_per_annum = self.periods_per_annum,
                    sharpe_ratio = 'TODO',
                )
            )

        # Return the results
        return efficient_frontier
'''


def test():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
