import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm


def data_load(path="namibia_raw_data.csv"):
    # read in
    df = pd.read_csv(path, low_memory=False)

    df.rename(
        columns={
            "q1": "q01",
            "q2": "q02",
            "q3": "q03",
            "qn6": "qn06",
            "qn7": "qn07",
            "qn8": "qn08",
            "qn9": "qn09",
        },
        inplace=True,
    )

    df = df[
        [
            "nation",
            "q01",
            "q02",
            "q03",
            "qn06",
            "qn07",
            "qn08",
            "qn09",
            "qn10",
            "qn11",
            "qn12",
            "qn13",
            "qn14",
            "qn15",
            "qn16",
            "qn17",
            "qn18",
            "qn19",
            "qn20",
            "qn21",
            "qn22",
            "qn23",
            "qn24",
            "qn25",
            "qn26",
            "qn27",
            "qn28",
            "qn29",
            "qn30",
            "qn31",
            "qn32",
            "qn33",
            "qn34",
            "qn35",
            "qn36",
            "qn37",
            "qn38",
            "qn39",
            "qn40",
            "qn41",
            "qn42",
            "qn43",
            "qn44",
            "qn45",
            "qn46",
            "qn47",
            "qn48",
            "qn49",
            "qn50",
            "qn51",
            "qn52",
            "qn53",
            "qn54",
            "qn55",
            "qn56",
            "qn57",
            "qn58",
        ]
    ]

    return df


def drop_and_slice(df, thresh=0.9):
    # reduce to columns with percent missing values below threshold
    cols = np.where((df.isna().sum() / df.shape[0]) < thresh)
    df = df.copy()[sorted(df.columns[cols])]

    # demographic columns
    country = df.copy().iloc[:, 0]
    demographics = df.copy().iloc[:, 1:4]
    demographics = demographics.astype("float64")
    demographics = demographics.fillna(-1)
    demographics[demographics == -1] = np.nan
    demographics = demographics.astype("float64")
    demographics.replace([np.inf, -np.inf], np.nan, inplace=True)

    # slice qn feature columns and rescale to binary 0/1
    colnames = df.columns[4:]
    features = df.copy()[colnames]
    features = features.astype("float64")
    features = features.fillna(-1)
    features[features == -1] = np.nan
    features = features.astype("float64")
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # adjust numbering on dichotomized variables
    # 1 = at risk
    # 2 = not at risk - convert to zero
    features[features == 2] = 0

    df = pd.DataFrame(country).join(demographics).join(features)

    return df


def reverse_binary(df, columns=None):
    # reverse 0/1 on the specified questions representing if at risk
    if columns is None:
        columns = [
            "qn07",
            "qn08",
            "qn47",
            "qn48",
            "qn49",
            "qn51",
            "qn54",
            "qn55",
            "qn56",
            "qn57",
            "qn58",
        ]
    for col in columns:
        zero_index = np.where(df[col] == 0)
        one_index = np.where(df[col] == 1)

        df[col].iloc[zero_index] = 1
        df[col].iloc[one_index] = 0

    return df


def min_max_scaler(df, columns=None):
    # normalize to 0-1 range
    if columns is None:
        columns = [
            "q01",
            "q02",
            "q03",
        ]
    for col in columns:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df


def impute(df):
    country = df.copy().iloc[:, 0]
    demographics = df.copy().iloc[:, 1:4]
    colnames = df.columns[4:]
    features = df.copy()[colnames]

    # DEMOGRAPHICS
    # impute demographics with rounded mean
    estimator = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=1)
    demographics_imp = np.round(estimator.fit_transform(demographics.join(features)))
    demographics_imp = demographics_imp[:, 0:3]
    demographics_imp = pd.DataFrame(demographics_imp, columns=demographics.columns)

    demographics_imp["nation"] = country.iloc[0]

    # MEDIAN
    # impute features with median
    estimator = SimpleImputer(missing_values=np.nan, strategy="median", verbose=1)
    # estimator.fit(features)

    # run imputation
    features_median = np.round(estimator.fit_transform(features))

    # imputation returns missing columns where there was no data
    # apply the correct column headers then stack the missing columns back on
    features_median = pd.DataFrame(
        features_median, columns=features.columns[np.where(features.sum() > 0)]
    )
    if len(features.columns[np.where(features.sum() == 0)]) > 0:
        features_median = features_median.join(
            pd.DataFrame(columns=features.columns[np.where(features.sum() == 0)])
        )
        features_median = features_median[sorted(features_median.columns)]
    else:
        features_median = features_median[sorted(features_median.columns)]

    # join demographics back in
    features_median = features_median.join(demographics_imp)
    features_median["imputation_method"] = "Simple_Median"

    # BAYES
    # impute features with bayesian ridge
    # setup estimator and fit data
    estimator = IterativeImputer(
        random_state=0,
        estimator=BayesianRidge(),
        max_iter=10,
        sample_posterior=True,
        min_value=0,
        max_value=1,
        verbose=1,
    )
    # estimator.fit(features)

    # run imputation
    features_bayes = np.round(estimator.fit_transform(features))

    # imputation returns missing columns where there was no data
    # apply the correct column headers then stack the missing columns back on
    features_bayes = pd.DataFrame(
        features_bayes, columns=features.columns[np.where(features.sum() > 0)]
    )
    if len(features.columns[np.where(features.sum() == 0)]) > 0:
        features_bayes = features_bayes.join(
            pd.DataFrame(columns=features.columns[np.where(features.sum() == 0)])
        )
        features_bayes = features_bayes[sorted(features_bayes.columns)]
    else:
        features_bayes = features_bayes[sorted(features_bayes.columns)]

    # join demographics back in
    features_bayes = features_bayes.join(demographics_imp)
    features_bayes["imputation_method"] = "Iterative_Bayes"

    # RANDOM FOREST
    # impute features with random forest
    estimator = IterativeImputer(
        random_state=0,
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=0),
        max_iter=10,
        min_value=0,
        max_value=1,
        verbose=1,
    )

    # run imputation
    features_trees = np.round(estimator.fit_transform(features))

    # imputation returns missing columns where there was no data
    # apply the correct column headers then stack the missing columns back on
    features_trees = pd.DataFrame(
        features_trees, columns=features.columns[np.where(features.sum() > 0)]
    )
    if len(features.columns[np.where(features.sum() == 0)]) > 0:
        features_trees = features_trees.join(
            pd.DataFrame(columns=features.columns[np.where(features.sum() == 0)])
        )
        features_trees = features_trees[sorted(features_trees.columns)]
    else:
        features_bayes = features_bayes[sorted(features_bayes.columns)]

        # join demographics back in
    features_trees = features_trees.join(demographics_imp)
    features_trees["imputation_method"] = "Iterative_Trees"

    # combine all imputed data
    combined_data = pd.concat([features_median, features_bayes, features_trees])

    return combined_data


def run_imputation(df):
    # empty frame to collect values
    all_data = pd.DataFrame()

    # iterate through countries, imputing individually
    for nation in tqdm(df["nation"].unique(), total=len(df["nation"].unique())):
        print("start country = ", nation)

        # run imputation
        combined_data = impute(df[df["nation"] == nation])

        # collect results
        all_data = pd.concat([all_data, combined_data])

    return all_data


if __name__ == "__main__":
    df = data_load(path="namibia_raw_data.csv")
    df = drop_and_slice(df, thresh=0.9)
    df = reverse_binary(df)
    df.to_csv("reduced_namibia_raw_data.csv", index=False)

    # country_list = ['Afghanistan', 'Algeria', 'Anguilla', 'Antigua and Barbuda']
    # df = df[df['nation'].isin(country_list)]
    
    all_data = run_imputation(df)
    #all_data = min_max_scaler(all_data)
    all_data.to_csv("namibia_imputed_data.csv")
