import pandas as pd
from IPython.display import display

from common.df_operations import separate_to_dict, \
    mapping
from common.mapping import n_days_average, \
    n_days_ago
from common.feature_set import FeatureSet

DATA_PATH = 'data/data.csv'


def run():
    # read data
    df = pd.read_csv(DATA_PATH)

    # get column names
    date_column = 'SYNC_DATE'
    quantity_column = 'OH_QTY'

    # change data's date columns to df format
    df = df.astype({date_column: 'datetime64[ns]'})

    product_df_dict = separate_to_dict(df, 'PRODUCT')
    # {'1': df of 1,
    #  '2': df of 2,
    # }
    # create lists of mapping functions
    print('two_day_average_feature_set\n')
    two_day_average_feature_set = train_two_days_averages_set(date_column, quantity_column, product_df_dict)
    two_day_average_feature_set.show('5')
    two_day_average_feature_set.show('6')

    print('two_days_feature_set\n')
    two_days_feature_set = train_two_days_set(date_column, quantity_column, product_df_dict)
    two_days_feature_set.show('5')
    two_days_feature_set.show('6')
    two_days_feature_set.show_summary()


def train_two_days_averages_set(date_column, quantity_column, product_df_dict):

    # create lists of mapping functions
    two_days_average_kwargs = {
        'date_column_name': date_column,
        'value_column_name': quantity_column,
        'n': 2
    }

    two_day_average_mapping_list = [{'function': n_days_average,
                                     'kwargs': two_days_average_kwargs}]

    # create feature sets
    two_day_average_feature_set = FeatureSet('two_day_average',
                                             product_df_dict,
                                             two_day_average_mapping_list
                                             )

    # start feature set mapping
    two_day_average_feature_set.map()
    two_day_average_feature_set.train_and_predict(quantity_column, [date_column])

    return two_day_average_feature_set


def train_two_days_set(date_column, quantity_column, product_df_dict):
    # create lists of mapping functions
    one_day_ago_kwargs = {
        'date_column_name': date_column,
        'value_column_name': quantity_column,
        'n': 1
    }
    two_days_ago_kwargs = {
        'date_column_name': date_column,
        'value_column_name': quantity_column,
        'n': 2
    }
    two_days_mapping_list = [{'function': n_days_ago,
                              'kwargs': one_day_ago_kwargs},
                             {'function': n_days_ago,
                              'kwargs': two_days_ago_kwargs}]
    # create feature sets
    two_days_feature_set = FeatureSet('two_days',
                                      product_df_dict,
                                      two_days_mapping_list
                                      )
    # start feature set mapping
    two_days_feature_set.map()
    two_days_feature_set.train_and_predict(quantity_column, [date_column])

    return two_days_feature_set


if __name__ == "__main__":
    run()
