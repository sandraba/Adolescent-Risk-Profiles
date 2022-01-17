import numpy as np
import pandas as pd

def data_load(path="scraped_data.csv"):
    # read in
    df = pd.read_csv(path, low_memory=False)

    # drop non-feature columns
    df = df.drop(
        [
            "Unnamed: 0",
            "id",
            "idno",
            "year_end",
            "file_title",
            "file-id",
            "href",
            "classid",
            "psu",
        ],
        axis=1,
    )    
    
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

if __name__ == "__main__":
    df = data_load(path="scraped_data.csv")
    (df[df["nation"]=="Namibia"]).to_csv("namibia_raw_data.csv")