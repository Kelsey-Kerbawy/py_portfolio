import pandas as pd
import numpy as np

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.black_litterman import market_implied_risk_aversion
from pypfopt.expected_returns import prices_from_returns


class Filtration:

    def significance_by_beta(
            regression_df: pd.DataFrame,
            min_beta: float,
            max_beta: float):

        selection_df = regression_df.loc[regression_df["beta_t_stat"].abs(
        ) > 2.0].loc[regression_df["betav"] > min_beta].loc[regression_df["betav"] < max_beta].iloc[:-1]

        if selection_df.empty:
            print(
                f'No securities had betas between {min_beta} and {max_beta}. Adjusting')
            selection_df = regression_df.loc[regression_df["beta_t_stat"].abs()
                                             > 2.0]

        if selection_df.empty:
            print(
                f'No securities had betas between {min_beta} and {max_beta} and significant t-statistic. Suggest restarting from watchlist')
            selection_df = regression_df

        return selection_df

    def significance_by_alpha(regress_stats_df, min_alpha):

        selection_df = regress_stats_df.loc[regress_stats_df["alpha_t_stat"].abs(
        ) > 2.0].loc[regress_stats_df["alpha"] > min_alpha].iloc[:-1]

        if selection_df.empty:
            print(f'No alpha greater than {min_alpha}.. Adjusting')

            selection_df = regress_stats_df.loc[regress_stats_df["alpha"]
                                                >= 0].iloc[:-1]

        if selection_df.empty:
            print(
                f'No securities had alphas greater than {min_alpha} and a significant t-statistic. Suggest restarting from watchlist.')

            selection_df = regress_stats_df

        return selection_df


class Allocation:

    def minimum_variance(
            expected_return_df: pd.DataFrame,
            cov_matrix_df: pd.DataFrame):

        expected_return_arr = expected_return_df.to_numpy().reshape(-1, 1)

        cov_matrix = cov_matrix_df.to_numpy()

        inv_cov_matrix = np.linalg.inv(cov_matrix)

        i = np.ones((expected_return_arr.shape))

        i_prime = np.transpose(i)

        weights = np.divide(np.matmul(inv_cov_matrix, i), np.matmul(
            np.matmul(i_prime, inv_cov_matrix), i))

        position_df = pd.DataFrame(
            index=list(expected_return_df.index.values),
            columns=['weights'],
            data=weights.tolist()
        )

        return position_df

    def tangency(
            expected_return_df: pd.DataFrame,
            cov_matrix_df: pd.DataFrame):

        expected_return_arr = expected_return_df.to_numpy().reshape(-1, 1)

        cov_matrix = cov_matrix_df.to_numpy()

        inv_cov_matrix = np.linalg.inv(cov_matrix)

        i = np.ones((expected_return_arr.shape))

        i_prime = np.transpose(i)

        weights = np.divide(
            np.matmul(
                inv_cov_matrix,
                expected_return_arr),
            np.matmul(
                np.matmul(
                    i_prime,
                    inv_cov_matrix),
                expected_return_arr))

        position_df = pd.DataFrame(
            index=list(expected_return_df.index.values),
            columns=['weights'],
            data=weights.tolist()
        )

        return position_df

    def gamma_optimal_weight(
            expected_return_df: pd.DataFrame,
            cov_matrix_df: pd.DataFrame,
            market_returns_s: pd.Series):

        market_prices_s = prices_from_returns(market_returns_s)

        gamma = market_implied_risk_aversion(
            market_prices_s, frequency=252, risk_free_rate=0.0)

        expected_return_arr = expected_return_df.to_numpy().reshape(-1, 1)

        cov_matrix = cov_matrix_df.to_numpy()

        weights = EfficientFrontier(
            expected_return_arr,
            cov_matrix).max_quadratic_utility(
            risk_aversion=gamma)
        weights = list(weights.items())

        new_w = list()
        for ind in range(len(weights)):
            new_w.append(weights[ind][1])

        weights = np.array(new_w)

        position_df = pd.DataFrame(
            index=list(expected_return_df.index.values),
            columns=['weights'],
            data=weights.tolist()
        )

        return position_df

    def by_method(
            expected_return_df: pd.DataFrame,
            cov_matrix_df: pd.DataFrame,
            market_returns_s: pd.Series,
            method: str):

        if method == 'min':

            position_df = Allocation.minimum_variance(
                expected_return_df.predict, cov_matrix_df)

        elif method == 'tangency':

            position_df = Allocation.tangency(
                expected_return_df.predict, cov_matrix_df)

        elif method == 'gamma':

            position_df = Allocation.gamma_optimal_weight(
                expected_return_df.predict, cov_matrix_df, market_returns_s)

        return position_df

    def long_short(
            expected_return_df: pd.DataFrame,
            cov_matrix_df: pd.DataFrame,
            is_market_neutral: bool):

        expected_return_arr = expected_return_df.to_numpy().reshape(-1, 1)

        cov_matrix = cov_matrix_df.to_numpy()

        weights = EfficientFrontier(
            expected_return_arr,
            cov_matrix,
            weight_bounds=(
                -1,
                1)).efficient_return(
            target_return=expected_return_df.max(),
            market_neutral=True)

        weights = list(weights.items())

        new_w = []
        for ind in range(len(weights)):
            new_w.append(weights[ind][1])

        position_df = pd.DataFrame(
            index=list(expected_return_df.index.values),
            columns=['weights'],
            data=new_w
        )

        return position_df


class Momentum:

    def smart_beta_signal(comparison_df:pd.DataFrame, recent_period_return: float):

        is_rebalance_tactical = False

        if comparison_df.loc['betav', 'holding_timeframe'] > \
                comparison_df.loc['betaub', 'historical_timeframe'] \
                and recent_period_return <= 0:

            is_rebalance_tactical = True

        if comparison_df.loc['betav', 'holding_timeframe'] < \
                comparison_df.loc['betalb', 'historical_timeframe'] \
                and recent_period_return >= 0:

            is_rebalance_tactical = True

        return is_rebalance_tactical

    def smart_alpha_signal(comparison_df:pd.DataFrame, recent_period_return:float):

        is_rebalance_tactical = False

        if comparison_df.loc['alpha', 'holding_timeframe'] < \
                comparison_df.loc['alphalb', 'historical_timeframe'] \
                and recent_period_return <= 0:

            is_rebalance_tactical = True

        return is_rebalance_tactical

    def prediction_signal(comparison_df, recent_period_return):

        is_rebalance_tactical = False

        if recent_period_return < \
                comparison_df.loc['predict_lower', 'historical_timeframe']:

            is_rebalance_tactical = True

        return is_rebalance_tactical
