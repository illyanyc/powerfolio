import pandas as pd

# Fundamental data filter
def fundamental_data_query(tickers_df, stock_fundamentals_df, fundamental_indicator_keys):
    '''Returns a pd.DataFrame of fundamental data filtered by user input range
    
    ...
    
    Parameters
    ----------
    tickers_df : pd.DataFrame - dataframe to be processed, contains tickers
    fundamental_indicator_key : str() - keyword for fundamental indicator requested
    
        Fundamental indicator keys ->
        
        P/E Ratio : [pe_ratio]
        EPS (Earnings per Share) : [eps]
        Annual Dividend : [dividend]
        Beta (vs. S&P 500) : [beta]
        EBIDT : [ebidt]
        Quick Ratio : [quick_ratio]
        3 Year Revenue Growth : [rev_growth]
        Free Cash Flow : [cash_flow]
    
    lower_bound : int() or float() - lower bound for fundamental value filter, default = -1000000
    upper_bound : int() or float() - upper bound for fundamental value filter, default = 1000000
     
     
     
    Returns
    -------
    result_df : pd.DataFrame - dataframe with ticker and filtered fundamental data
    '''

    fund_indicators_dict = {
        'pe_ratio' : 'peNormalizedAnnual',
        'eps' : 'epsNormalizedAnnual',
        'dividend' : 'dividendsPerShareTTM',
        'beta' : 'beta',
        'ebidt' : 'ebitdPerShareTTM',
        'quick_ratio' : 'quickRatioAnnual',
        'rev_growth' : 'revenueGrowth3Y',
        'free_cash_flow' : 'freeCashFlowAnnual'   
    }

    # Declare result_df
    result_df = pd.DataFrame()
    
    # Declate tickers list
    tickers_list = tickers_df['Symbol']
    
    # Declare fundamental data df and filter by ticker df
    data_df = stock_fundamentals_df[stock_fundamentals_df.symbol.isin(tickers_list)]
    data_df = data_df.set_index(['symbol'])
    
    # Extract requested fundamental data
    for ind in fundamental_indicator_keys:
        df = data_df[data_df['metric_type'] == fund_indicators_dict[ind]]
        result_df = pd.concat([result_df, df], axis = 1, join = 'outer')
    
        # Clean up df
        result_df = result_df.drop(columns = ['metric_type', 'series'])
        result_df = result_df.rename(columns = {'symbol' : 'ticker', 
                                'metric' : ind})
        
            # Convert all df values to numeric
        result_df[ind] = result_df[ind].apply(pd.to_numeric)

    
    
    return result_df