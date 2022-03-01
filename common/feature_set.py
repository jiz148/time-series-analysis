"""
Feature Class
"""
from sklearn import linear_model
import sklearn.metrics as metrics
from IPython.display import display

from common.df_operations import mapping


class FeatureSet:

    def __init__(self, name, df_dict, mapping_list):
        """
        @param mapping_list: needs to be in the form like [{function: ..., kwargs: ...},{function: ..., kwargs: ...}]
        """
        self.name = name
        self.df_dict = df_dict.copy()
        self.mapping_list = mapping_list

    def map(self):
        for key in self.df_dict.keys():
            for m in self.mapping_list:
                self.df_dict[key] = mapping(self.df_dict[key], m['function'],  **m['kwargs'])

    def train_and_predict(self, predict_column, ignore_columns):
        """
        Train for each key of dataframes
        result sample
        {
            '1': {
                    'df': df of product '1',
                    'stats': {
                            'r2': value,
                            'std': value,
                            'super stat': value,
                        }
                }
        }
        object.df_dict['5']['df']
        object.df_dict['5']['stats']['r2']
        """
        for key in self.df_dict.keys():
            df = self.df_dict[key]
            x_cols = list(df.columns.copy())
            x_cols.remove(predict_column)
            for ignore_col in ignore_columns:
                x_cols.remove(ignore_col)
            # separate data
            x = df[x_cols]
            y = df[predict_column]
            # train
            regressor = linear_model.LinearRegression()
            regressor.fit(x, y)
            y_hat = regressor.predict(x)
            df['regr_result'] = y_hat
            # save stats
            self.df_dict[key] = {
                'df': df,
                'stats': {
                    'r2': metrics.r2_score(y, y_hat),
                },
            }

    def show(self, key):
        display(self.df_dict[key])

    def show_summary(self):
        """
        @return:
        product_id, r2, ...
        1           .9, ...
        """
        for key in self.df_dict:
            print(self.df_dict[key]['stats']['r2'])
