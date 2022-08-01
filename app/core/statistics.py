import pandas as pd


def population_stats(returns_df: pd.DataFrame):

    pop_stats_df = pd.DataFrame(
        {
            'average': returns_df.mean(
                axis=0),
            'stdev': returns_df.std(
                axis=0),
            'variance': returns_df.var(
                axis=0),
            'skewness': returns_df.skew(
                axis=0),
            'kurtoses': returns_df.kurtosis(
                axis=0),
            'confub': returns_df.mean(
                axis=0) +
            1.96 *
            returns_df.std(
                axis=0),
            'conflb': returns_df.mean(
                axis=0) -
            1.96 *
            returns_df.std(
                axis=0)})

    pop_stats_df.index.name = 'tickers'

    return pop_stats_df


def sample_stats(returns_df: pd.DataFrame, sample_start: str):

    pop_stats_df = population_stats(returns_df)

    sample_stats_df = pd.DataFrame({'average': returns_df.loc[sample_start:].mean(axis=0),
                                    'ster': returns_df.loc[sample_start:].sem(axis=0, ddof=1),
                                    'skewness': returns_df.loc[sample_start:].skew(axis=0),
                                    'kurtoses': returns_df.loc[sample_start:].kurtosis(axis=0),
                                    'confub': returns_df.loc[sample_start:].mean(axis=0) + 1.96 * returns_df.loc[sample_start:].sem(axis=0, ddof=1),
                                    'conflb': returns_df.loc[sample_start:].mean(axis=0) - 1.96 * returns_df.loc[sample_start:].sem(axis=0, ddof=1)
                                    })
    sample_stats_df.index.name = 'tickers'

    t_stat = []
    value_zip = zip(
        list(
            pop_stats_df.average.values), list(
            sample_stats_df.average.values), list(
            sample_stats_df.ster.values))

    for pop_list, samp_list, ster_list in value_zip:
        t_stat.append((samp_list - pop_list) / ster_list)

    sample_stats_df.insert(len(sample_stats_df.columns), 't_stat', t_stat)

    return sample_stats_df
