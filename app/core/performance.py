import pandas as pd
import numpy as np
import statsmodels.api as sm

from core.regression import regression
from core.prediction import regressed_expected_returns


def cumulative_returns(returns_df: pd.DataFrame):
    arr = np.cumprod(np.array(returns_df + 1), axis=0)

    returns_df = pd.DataFrame(data=arr, columns=list(
        returns_df.columns.values), index=list(returns_df.index.values))

    returns_df.index.name = 'Date'

    return returns_df


def hypothesis_test(
        sample_stats_df: pd.DataFrame,
        population_stats_df: pd.DataFrame):

    if len(sample_stats_df) != len(population_stats_df):

        print("population dataframe and sample dataframe not of same length. Cannot conduct hypothesis test.")
        quit()

    hypo_list = []
    for row in range(len(sample_stats_df.index.values)):

        reject_sample_average = sample_stats_df.iloc[row].loc["t_stat"] >= 2

        if sample_stats_df.iloc[row].loc["confub"] <= population_stats_df.iloc[row].loc["average"]:
            reject_sample_average = True
        elif sample_stats_df.iloc[row].loc["conflb"] >= population_stats_df.iloc[row].loc["average"]:
            reject_sample_average = True
        else:
            reject_sample_average = False

        hypo_list.append(reject_sample_average)

    hypothesis_df = pd.DataFrame(
        data=hypo_list,
        index=list(
            sample_stats_df.index.values),
        columns=['samplereject'],
        dtype=None,
        copy=False)

    return hypothesis_df


def regression_comparison(
        historical_returns_df: pd.DataFrame,
        holding_returns_df: pd.DataFrame,
        market_var: str):

    ex_ante_returns_df = historical_returns_df.loc[historical_returns_df.index < list(
        holding_returns_df.index.values)[0]]

    historical_regression_df = regression(
        historical_returns_df,
        market_var).drop(market_var).rename(
        index={
            'position': 'historical_timeframe'})
    holding_regression_df = regression(holding_returns_df, market_var).drop(
        market_var).rename(index={'position': 'holding_timeframe'})
    ex_ante_regression_df = regression(ex_ante_returns_df, market_var).drop(
        market_var).rename(index={'position': 'before_purchase'})

    comparison_df = pd.concat(
        [historical_regression_df.T, holding_regression_df.T, ex_ante_regression_df.T], axis=1)

    recent_period_return = holding_returns_df.drop(
        columns=market_var).values.tolist()[-1][-1]

    comparison_dct = {
        'comparison_df': comparison_df,
        'recent_period_return': recent_period_return
    }
    return comparison_dct


def prediction_comparison(
        historical_returns_df: pd.DataFrame,
        holding_returns_df: pd.DataFrame,
        market_var: str):

    ex_ante_returns_df = historical_returns_df.loc[historical_returns_df.index < list(
        holding_returns_df.index.values)[0]]

    historical_prediction_df = regressed_expected_returns(
        historical_returns_df,
        market_var).drop(market_var).rename(
        index={
            'position': 'historical_timeframe'})
    holding_prediction_df = regressed_expected_returns(
        holding_returns_df,
        market_var).drop(market_var).rename(
        index={
            'position': 'holding_timeframe'})
    ex_ante_prediction_df = regressed_expected_returns(
        ex_ante_returns_df,
        market_var).drop(market_var).rename(
        index={
            'position': 'before_purchase'})

    comparison_df = pd.concat(
        [historical_prediction_df.T, holding_prediction_df.T, ex_ante_prediction_df.T], axis=1)

    recent_period_return = holding_returns_df.drop(
        columns=market_var).values.tolist()[-1][-1]

    comparison_dct = {
        'comparison_df': comparison_df,
        'recent_period_return': recent_period_return
    }
    return comparison_dct


def TM_timing(
        holding_returns_df: pd.DataFrame,
        market_var: str):

    x = holding_returns_df[market_var].values

    x_with_const = sm.add_constant(x.reshape(-1, 1))

    data = {'a': holding_returns_df['position'].values[0:], 'b': x_with_const}
    model = sm.OLS.from_formula('a ~ b + np.power(b,2)', data=data)
    result = model.fit()

    return result.params[4]
