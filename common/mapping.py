"""
Mapping functions for Pandas df
"""
import pandas as pd
from IPython.display import display

from common.df_operations import mapping


def n_days_average(df, date_column_name, value_column_name, n):
    """
    Add column n days average, excluding current row
    """
    # insert missing date
    df = _insert_missing_date(df, date_column_name)
    # insert column
    column_name = str(n) + '_days_average'
    df[column_name] = df[value_column_name].rolling(n, closed="left", min_periods=0).mean()
    # change first row to current value
    df.iloc[0, df.columns.get_loc(column_name)] = df.iloc[0, df.columns.get_loc(value_column_name)]
    return df


def n_days_ago(df, date_column_name, value_column_name, n):
    """
    Add column n days ago
    """
    # insert missing date
    df = _insert_missing_date(df, date_column_name)
    # insert column
    column_name = str(n) + '_day(s)_ago'
    df[column_name] = df[value_column_name].shift(n)
    # fill na with ahead values
    df = df.fillna(method='bfill')
    df[column_name].fillna(df[value_column_name], inplace=True)
    return df


def _insert_missing_date(df, date_column_name):
    """
    Insert Missing date to df
    """
    # insert missing date
    df = df.sort_values(by=date_column_name)
    all_dates = pd.DataFrame(pd.date_range(df[date_column_name].min(),
                                           df[date_column_name].max()),
                             columns=[date_column_name])
    # from the all_dates DataFrame, left join onto the DataFrame with missing dates
    df = all_dates.merge(right=df, how='left', on=date_column_name).fillna(method='ffill')
    return df


if __name__ == "__main__":
    df = pd.DataFrame([['06/10/2020', 25],
                       ['06/14/2020', 13],
                       ['06/15/2020', 2],
                       ['06/19/2020', 245],
                       ['06/23/2020', 215]], columns=['date', 'qt'])
    df = df.astype({'date': 'datetime64[ns]'})
    # display(n_days_average(df, 'date', 'qt', 2))
    display(mapping(df, n_days_ago, date_column_name='date', value_column_name='qt', n=4))
