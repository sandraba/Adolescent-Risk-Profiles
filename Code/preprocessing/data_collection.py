import time

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

# from selenium.webdriver.firefox.options import Options
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm


def gshs_catalogue():

    # list of all 151 studies in the global school based student health survey
    list_data_link = "https://extranet.who.int/ncdsmicrodata/index.php/catalog/export/csv?ps=5000&collection[]=GSHS"
    list_data = pd.read_csv(list_data_link)

    # keep only the most recent survey
    latest_list_data = (
        list_data.copy()
        .sort_values(by=["nation", "year_end"], ascending=[True, False])
        .drop_duplicates(subset=["nation"])
    )

    return latest_list_data


def scrape_links(latest_list_data):
    # selenium to access each page
    # the microdata page has a form submit that's required to get the data link
    fireFoxOptions = Options()
    fireFoxOptions.headless = True
    fireFoxOptions.add_argument("--no-sandbox")
    fireFoxOptions.add_argument("--disable-dev-shm-usage")
    fireFoxOptions.add_argument("--disable-extensions")
    driver = webdriver.Chrome("/Users/sandrabee/Seafile/HIGHsea/HIGH/Papers/__In_Progress/Risk_Profiles/ARP_Namibia/chromedriver", options=fireFoxOptions)

    # collect all download link information
    download_links = []
    # iterate through country ids
    k = 0
    for i in tqdm(latest_list_data["id"]):
        # connection to country page
        driver.get(
            "https://extranet.who.int/ncdsmicrodata/index.php/catalog/{}/get-microdata".format(
                i
            )
        )
        # sleep allows the page to load properly first
        time.sleep(3)
        # submit the terms and conditions accept
        driver.find_element_by_xpath('//*[@type="submit"]').click()
        # pass response to beautiful soup
        soup = BeautifulSoup(driver.page_source, features="lxml")
        # cycle through results until reaching the top csv resource
        for j in range(len(soup.find_all(class_="resource-left-col"))):
            if (
                soup.find_all(class_="resource-left-col")[j]
                .find(class_="download")
                .attrs["data-extension"]
                == "csv"
            ) and (
                "National"
                in str(
                    soup.find_all(class_="resource-left-col")[j].find(
                        class_="resource-info"
                    )
                )
            ):
                k = j
                break
            else:
                k = 0
        # find all download classes and extract download link for top listed dataset
        top_resource = soup.find_all(class_="resource-left-col")[k]
        # pull resource information
        download_links.append(
            {  # reference list id for table join
                "list-id": i,
                # first file title from div-span
                "title": top_resource.find(class_="resource-info")
                .contents[4]
                .replace("\n", "")
                .replace("\t", "")
                .strip(),
                # file id
                "file-id": top_resource.find(class_="resource-info").attrs["id"],
                # download link
                "href": top_resource.contents[3].find(class_="download").attrs["href"],
            }
        )

    return download_links


def combine_lists(latest_list_data, download_links):
    # merge, remove and rename columns
    list_links = pd.merge(
        left=latest_list_data,
        right=pd.DataFrame(download_links),
        left_on="id",
        right_on="list-id",
        how="left",
    )
    list_links = list_links[
        ["id", "idno", "nation", "year_end", "title_y", "file-id", "href"]
    ]
    list_links.rename(columns={"title_y": "file_title"}, inplace=True)

    # exclude problem links
    problem_links = list(
        list_links[list_links["file_title"] != "National dataset (csv)"]["id"]
    )
    csv_list = list_links[~list_links["id"].isin(problem_links)]

    # empty objects to collect values
    col_headers = []

    # iterate through dataframe rows
    for index, row in tqdm(csv_list.iterrows(), total=csv_list.shape[0]):
        returned_headers = pd.read_csv(row["href"], nrows=0).columns.tolist()
        temp_headers = [i.lower() for i in returned_headers]
        col_headers += temp_headers

    header_set = set(col_headers)

    return header_set, csv_list


def csv_downloads(header_set, csv_list):
    # empty objects to collect values
    combined_data = pd.DataFrame(columns=sorted(header_set))

    # iterate through dataframe rows
    for index, row in tqdm(csv_list.iterrows(), total=csv_list.shape[0]):
        # download csv
        temp_csv = pd.read_csv(row["href"])

        # lower, replace and store column headers
        temp_headers = [i.lower() for i in temp_csv.columns]
        temp_csv.columns = temp_headers

        # append file id for table joins
        temp_csv["file-id"] = row["file-id"]

        # merge
        combined_data = pd.concat([combined_data, temp_csv])

    all_data = pd.merge(left=csv_list, right=combined_data, on="file-id", how="left")

    return all_data


if __name__ == "__main__":
    latest_list_data = gshs_catalogue()
    download_links = scrape_links(latest_list_data)
    header_set, csv_list = combine_lists(latest_list_data, download_links)
    all_data = csv_downloads(header_set, csv_list)
    all_data.to_csv("scraped_data.csv")