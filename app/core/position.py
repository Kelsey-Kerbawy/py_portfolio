import pandas as pd
from data_capture import FromYF
from pypfopt.expected_returns import returns_from_prices


def position_historical_returns(
        market_ticker: str,
        tickers: list,
        actual_weights: list, period: str = 'max'):

    historical_prices_df = FromYF.get_prices(
        tickers,
        period=period,
        interval='1d',
        is_adjusted=True)

    historical_prices_s = historical_prices_df.dot(actual_weights)
    historical_prices_s.name = 'position'
    historical_prices_df = historical_prices_s.to_frame()
    historical_market_prices = FromYF.get_prices(
        market_ticker,
        start=pd.to_datetime(
            list(
                historical_prices_df.index.values)[0]).strftime("%Y-%m-%d"),
        end=pd.to_datetime(
            list(
                historical_prices_df.index.values)[
                -1]).strftime("%Y-%m-%d"),
        interval='1d',
        is_adjusted=True)
    historical_market_prices.name = market_ticker
    historical_market_prices_df = historical_market_prices.to_frame()
    historical_prices_df = pd.concat([historical_prices_df, historical_market_prices_df], axis=1)
    historical_returns_df = returns_from_prices(historical_prices_df)
    historical_returns_df.index = historical_returns_df.index.strftime("%Y-%m-%d")

    return historical_returns_df


def position_holding_returns(
    market_ticker: str,
    tickers: list, actual_weights: list,
    purchase_date: str
):

    holding_prices_df = FromYF.get_prices(tickers, start=purchase_date, end=pd.Timestamp.now(
        tz='America/Detroit').round(freq='d').strftime("%Y-%m-%d"), interval='1d', is_adjusted=True)

    holding_prices_s = holding_prices_df.dot(actual_weights)
    holding_prices_s.name = 'position'
    holding_prices_df = holding_prices_s.to_frame()
    holding_market_prices = FromYF.get_prices(
        market_ticker, start=purchase_date, end=pd.Timestamp.now(
            tz='America/Detroit').round(freq='d').strftime("%Y-%m-%d"), interval='1d', is_adjusted=True)

    holding_market_prices.name = market_ticker
    holding_market_prices_df = holding_market_prices.to_frame()
    holding_prices_df = pd.concat(
        [holding_prices_df, holding_market_prices_df], axis=1)

    holding_returns_df = returns_from_prices(holding_prices_df)

    holding_returns_df.index = holding_returns_df.index.strftime("%Y-%m-%d")

    return holding_returns_df
