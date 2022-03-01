"""
Pandas df operations
"""
import pandas as pd


def separate_to_dict(df, separator):
    """
    Separate df to dict list like {'product_id': df of product 1 with no separator column}
    """
    unique_separators = df[separator].unique()
    # create a data frame dictionary to store your data frames
    df_dict = {str(elem): pd.DataFrame for elem in unique_separators}

    for key in df_dict.keys():
        df_dict[key] = df[:][df[separator] == int(key)].drop(separator, axis=1).reset_index(drop=True)

    return df_dict


def mapping(df, mapping_function, **kwargs):
    """
    Gets **kwargs and do mapping with the input mapping_function
    """
    return mapping_function(df, **kwargs)
