import asyncio
import datetime

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from .data_collection import gshs_catalogue, scrape_links, combine_lists, csv_downloads
from .data_imputation import (
    drop_and_slice,
    data_load,
    reverse_binary,
    run_imputation,
    min_max_scaler,
)

def preprocess_data(df: pd.DataFrame, thresh: float = 0.5, count=None):
    # lists of non-factor columns
    imputation_method = df["imputation_method"]
    row_key = df[["Unnamed: 0"]].rename(columns={"Unnamed: 0": "row_key"})
    nation = df[["nation"]]

    # drop non-factor columns and set data format
    df = df.drop(
        ["Unnamed: 0", "imputation_method", "nation"],
        axis=1,
        errors="ignore",  # will only drop if exists
    )
    if count is not None:
        df = df.iloc[0:count].astype("float16")
    else:
        df = df.astype("float16")

    for col in df.columns:
        max_value = df.loc[df[col] != np.inf, col].max()
        df[col].replace(np.inf, max_value, inplace=True)
        df[col].replace(np.nan, max_value, inplace=True)

    # drop na columns
    df = df.copy()[df.columns[np.where(df.sum() > 0)]]

    # bring imputation method back in
    df["imputation_method"] = imputation_method
    df["nation"] = nation

    return df, row_key.join(nation).join(imputation_method)


def process_data(logger=None):
    print("Getting latest catalogue...")
    latest_list_data = gshs_catalogue()
    print("Scraping links...")
    download_links = scrape_links(latest_list_data)
    header_set, csv_list = combine_lists(latest_list_data, download_links)
    del download_links
    del latest_list_data
    print("Downloading CSVs...")
    all_data = csv_downloads(header_set, csv_list)
    all_data.to_csv("scraped_data.csv")
    del all_data
    print("Cleaning scraped data...")
    df = data_load(path="scraped_data.csv")
    df = drop_and_slice(df, thresh=0.9)
    df = reverse_binary(df)
    all_data = run_imputation(df)
    all_data = min_max_scaler(all_data)
    all_data.to_csv("namibia_raw_data.csv")
    print("Preprocess Data...")
    df, row_key = preprocess_data(pd.read_csv("namibia_raw_data.csv"))
    print("Final data size: {}".format(df.shape))
    return df, row_key

if __name__ == "__main__":
    ref_time = time.time()
    process_data()
    print(f"Time: {time.time()-ref_time:.2f}s")
    # df = pd.read_csv("imputed_data.csv")
    # df, row_key = preprocess_data(df)
    # print(df.shape)
    # # print(df.memory_usage(index=False, deep=False))
    # # print(df.memory_usage(index=False, deep=False).sum() / 1024 / 1024, "mb")
    # # print()
    # # for col in df.columns.values:
    # #     print(col, df[col].max())
    # sql_commands(df, row_key)
