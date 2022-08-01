import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from core.statistics import population_stats


def regressed_expected_returns(returns_df: pd.DataFrame, market_var: str):

    statistics_df = population_stats(returns_df)

    market_prediction = statistics_df.loc[market_var].average

    x_prediction_with_constant = sm.add_constant(
        np.array([market_prediction]), has_constant='add')

    x = returns_df[market_var].values

    x_with_const = sm.add_constant(x.reshape(-1, 1))

    E_dct = {'tickers': [],
             'predict': [],
             'predict_lower': [],
             'predict_upper': []}

    for column in returns_df:

        model = sm.OLS(
            (returns_df[column].values[0:]), x_with_const)

        result = model.fit()

        E_dct['tickers'].append(column)
        E_dct['predict'].append(result.predict(
            exog=x_prediction_with_constant)[0])
        E_dct['predict_lower'].append(result.get_prediction(
            exog=x_prediction_with_constant).conf_int(0.05)[0][0])
        E_dct['predict_upper'].append(result.get_prediction(
            exog=x_prediction_with_constant).conf_int(0.05)[0][1])

    E_df = pd.DataFrame.from_dict(E_dct)
    E_df.set_index('tickers', inplace=True)
    return E_df


def relative_shortfall_risk(returns_df: pd.DataFrame, periods: int):

    statistics_df = population_stats(returns_df)

    blank_matrix = pd.DataFrame(
        data=None,
        index=[
            "p(R<" +
            s +
            ")" for s in list(
                statistics_df.index.values)],
        columns=list(
            statistics_df.index.values),
        dtype=None,
        copy=False)

    count = 0
    countdos = 0

    prob_list_list = []

    for column in blank_matrix:
        prob_list = []

        for row in blank_matrix[column]:
            mean_diff = periods * \
                ((returns_df.iloc[:, countdos] - returns_df.iloc[:, count]).mean())
            var_diff = periods * \
                ((returns_df.iloc[:, countdos] - returns_df.iloc[:, count]).var())

            if var_diff == 0:
                # The probability that an asset will underperform itself is
                # NULL because you'll have to divide by zero, enter -1 in that
                # position to make it obvious.
                prob = 0
            else:
                # Formula for relative shortfall probability.
                prob = norm.cdf(-mean_diff / np.sqrt(var_diff))

            prob_list.append(prob)

            countdos = countdos + 1

        count = count + 1
        countdos = 0

        prob_list_list.append(prob_list)

    shortfall_df = pd.DataFrame(
        data=prob_list_list,
        index=[
            "p(<" +
            s +
            ")" for s in list(
                statistics_df.index.values)],
        columns=list(
            statistics_df.index.values),
        dtype=None,
        copy=False)

    return shortfall_df
