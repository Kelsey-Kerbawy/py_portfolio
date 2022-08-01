import pandas as pd
import numpy as np
import statsmodels.api as sm


def regression(returns_df: pd.DataFrame, market_var: str):

    x = returns_df[market_var].values

    x_with_const = sm.add_constant(x.reshape(-1, 1))

    regression_statistics_dct = {'tickers': [],
                                 'alpha': [],
                                 'betav': [],
                                 'alpha_t_stat': [],
                                 'beta_t_stat': [],
                                 'skewness': [],
                                 'kurtoses': [],
                                 'betaub': [],
                                 'betalb': [],
                                 'alphalb': [],
                                 'alphaub': [],
                                 'epsilon': []}

    for column in returns_df:

        model = sm.OLS((returns_df[column].values[0:]), x_with_const)

        result = model.fit()

        summary = result.summary()

        regression_statistics_dct['tickers'].append(column)
        regression_statistics_dct['alpha'].append(result.params[0])
        regression_statistics_dct['betav'].append(result.params[1])
        regression_statistics_dct['alpha_t_stat'].append(result.tvalues[0])
        regression_statistics_dct['beta_t_stat'].append(result.tvalues[1])
        regression_statistics_dct['skewness'].append(
            summary.tables[2].data[2][1])
        regression_statistics_dct['kurtoses'].append(
            summary.tables[2].data[3][1])
        regression_statistics_dct['betaub'].append(result.conf_int(0.05)[1][1])
        regression_statistics_dct['betalb'].append(result.conf_int(0.05)[1][0])
        regression_statistics_dct['alphaub'].append(
            result.conf_int(0.05)[0][1])
        regression_statistics_dct['alphalb'].append(
            result.conf_int(0.05)[0][0])
        regression_statistics_dct['epsilon'].append(np.sqrt(result.mse_model))

    regression_df = pd.DataFrame.from_dict(regression_statistics_dct)
    regression_df = regression_df.set_index('tickers')

    return regression_df
