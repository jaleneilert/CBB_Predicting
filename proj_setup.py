import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def read(filepath1, filepath2):
    # read csv file
    cbb_df = pd.read_csv(filepath1)
    cbb_25 = pd.read_csv(filepath2)

    # team needed to be capitalized to combine the two (in the cbb_25 dataset)
    cbb_df = pd.concat([cbb_df, cbb_25])

    # replace whitespace in team column for graphing
    cbb_df['TEAM'] = cbb_df['TEAM'].replace(' ', '_', regex = True)
    cbb_df.head()

    # Want to drop the seed because it tells us their ranking in the tournament
    cbb_df = cbb_df.drop('SEED', axis = 1)

    #print(cbb_df)

    cbb_df.dtypes

    cbb_df['POSTSEASON'].unique()

    # return the resulting data frame
    return cbb_df

def encode_postseason(cbb_df):
    # replace all appearance of not making the tournament with 0, else 1
    cbb_df['POSTSEASON'] = cbb_df['POSTSEASON'].astype(str)
    cbb_df['POSTSEASON'] = np.where(cbb_df['POSTSEASON'].isin(['NA', 'N/A', 'nan']), 0, 1)

    #print(cbb_df['POSTSEASON'].unique())
    #print(cbb_df['POSTSEASON'])

    # return modified data frame
    return cbb_df

def encode_conference(cbb_df):
    # Power Five conferences (SEC, ACC, B10, B12, BE) and then Mid-Major
    power_five = ['SEC', 'ACC', 'B10', 'B12', 'BE']
    cbb_df['CONF'] = cbb_df['CONF'].astype(str)
    cbb_df['CONF'] = np.where(cbb_df['CONF'].isin(power_five), cbb_df['CONF'], 'Mid_Major')

    cbb_df['CONF'].unique()

    cbb_df = cbb_df.drop('TEAM', axis=1)

    cbb_df.head()

    one_encoder = OneHotEncoder(sparse_output=False)

    encoded_data = one_encoder.fit_transform(cbb_df[['CONF']])

    one_df = pd.DataFrame(encoded_data, columns=one_encoder.get_feature_names_out(['CONF']))

    cbb_df = cbb_df.drop(['CONF'], axis=1)

    cbb_df = cbb_df.reset_index(drop=True)
    one_df = one_df.reset_index(drop=True)

    #update the existing data to include the one hot encoded conferences
    cbb_df = pd.concat([cbb_df, one_df], axis=1)

    # return resulting data frame
    #print(cbb_df)
    return cbb_df

# split the data into train (this will be split into validation in window) and test
def split_data(cbb_df):
    initial_train = cbb_df[(cbb_df['YEAR'] < 2024) & (cbb_df['YEAR'] != 2020)]
    initial_test = cbb_df[cbb_df['YEAR'] >= 2024]

    return initial_train, initial_test

def init_window(X, y, i):
    # need to account for missing 2021 data due to no postseason

    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]

    training_years = years[i: i + 3]
    validation_year = years[i + 3]

    # bool array telling us which data in X have the years we want
    train_m = X['YEAR'].isin(training_years)
    validation_m = X['YEAR'] == validation_year

    train_indices = np.where(train_m)[0]
    valid_indices = np.where(validation_m)[0]

    Xtrain = X.loc[train_indices]
    ytrain = y.loc[train_indices]
    Xvalid = X.loc[valid_indices]
    yvalid = y.loc[valid_indices]

    #print(f"training years: {training_years}")
    #print(f"Validation year: {validation_year}")
    #print(f"Y train length: {len(ytrain)}")
    #print(f"Y valid length: {len(yvalid)}")
    return  Xtrain, ytrain, Xvalid, yvalid