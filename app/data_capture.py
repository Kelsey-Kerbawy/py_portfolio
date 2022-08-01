import pandas as pd

import yfinance as yf


class FromYF:

    def get_asset(ticker: str):
        '''Information about a security'''
        asset_data = {
            'index': [
                'ticker',
                'company_name',
                'sector',
                'country'],
            'columns': ['info'],
            'data': [
                ticker,
                yf.Ticker(ticker).info['longName'],
                yf.Ticker(ticker).info['sector'],
                yf.Ticker(ticker).info['country']],
        }

        asset_df = pd.DataFrame(
            index=asset_data['index'],
            columns=asset_data['columns'],
            data=asset_data['data']
        )

        return asset_df

    def get_bar(ticker: str, period: str):

        bar_df = yf.Ticker(ticker).history(period)

        return bar_df

    def get_asset_dividends(ticker: str):

        dividend_s = yf.Ticker(ticker).dividends

        dividend_df = dividend_s.to_frame(name=ticker)

        return dividend_df

    def get_prices(
            tickers: list,
            period: str = None,
            start: str = None,
            end: str = None,
            interval: str = '1d',
            is_adjusted: bool = True):

        if period is not None:
            price_df = yf.download(
                tickers,
                auto_adjust=is_adjusted,
                period=period,
                interval=interval).Close.dropna(
                axis='index')

        if (start and end) is not None:
            price_df = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=is_adjusted,
                interval=interval).Close.dropna(
                axis='index')

        return price_df
