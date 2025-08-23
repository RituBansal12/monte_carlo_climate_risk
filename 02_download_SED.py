import requests
import re
import io
import gzip
import os
import argparse
from collections import defaultdict
import pandas as pd

BASE_URL_DEFAULT = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
CATEGORIES = ["details", "fatalities", "locations"]

# Pre-compiled regexes
DIR_HREF_RE = re.compile(r'href="([^\"]+\.csv\.gz)"')
# Example: StormEvents_details-ftp_v1.0_d1950_c20250520.csv.gz
FILE_RE = re.compile(r"StormEvents_([a-z]+)-.*?_d(\d{4})_c(\d{8}).*?\.csv\.gz", re.IGNORECASE)

def list_available_files(session, base_url):
    """
    Retrieve the HTML index and extract all filenames.

    Parameters
    ----------
    base_url : str
        URL to the NOAA Storm Events CSV directory.

    Returns
    -------
    list[str]
        Filenames found in the directory listing ('.csv.gz').
    """
    resp = session.get(base_url)
    resp.raise_for_status()
    html = resp.text
    # Regex to grab filenames ending in .csv.gz
    return DIR_HREF_RE.findall(html)

def filter_files(files, categories, start_year, end_year):
    """
    Filter file list by desired categories and year range, keeping the latest
    revision per (category, year).

    Parameters
    ----------
    files : list[str]
        All filenames from the directory listing.
    categories : list[str]
        Target categories (details, fatalities, locations).
    start_year : int
        Start year (inclusive).
    end_year : int
        End year (inclusive).

    Returns
    -------
    list[tuple]
        List of tuples (category, year, filename) with latest c-date per year.
    """
    latest = {}  # (cat, year) -> (cdate, fname)
    for fname in files:
        m = FILE_RE.search(fname)
        if not m:
            continue
        cat, year_s, cdate = m.group(1).lower(), m.group(2), m.group(3)
        year = int(year_s)
        if cat not in categories or not (start_year <= year <= end_year):
            continue
        key = (cat, year)
        prev = latest.get(key)
        if (prev is None) or (cdate > prev[0]):
            latest[key] = (cdate, fname)
    # Emit list
    out = []
    for (cat, year), (_cdate, fname) in sorted(latest.items()):
        out.append((cat, year, fname))
    return out

def download_and_parse(session, base_url, fname):
    """
    Download a gzip CSV file and return a pandas DataFrame.

    Parameters
    ----------
    base_url : str
        Base URL of the directory.
    fname : str
        File name to download ('.csv.gz').

    Returns
    -------
    pd.DataFrame
        Parsed CSV content.
    """
    url = f"{base_url}{fname}"
    resp = session.get(url)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content), mode='rt') as f:
        df = pd.read_csv(f, dtype=str, low_memory=False)
    return df

def save_parquet(category_dfs, category, start_year, end_year, output_dir="data"):
    """
    Concatenate and save a single Parquet per category under output_dir.

    Parameters
    ----------
    category_dfs : list[pd.DataFrame]
        DataFrames to concatenate.
    category : str
        Category name.
    start_year : int
        Start year (inclusive) for file naming.
    end_year : int
        End year (inclusive) for file naming.
    output_dir : str
        Target directory to write Parquet files (default: 'data').
    """
    df = pd.concat(category_dfs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"SED_{category}_{start_year}-{end_year}.parquet")
    df.to_parquet(out_file, index=False)
    print(f"Saved {out_file} with {len(df):,} rows")

def main(
    base_url=BASE_URL_DEFAULT,
    categories=CATEGORIES,
    start_year=1950,
    end_year=2025,
    output_dir="data",
):
    """
    Download and aggregate NOAA Storm Events data into category Parquet files.

    Parameters
    ----------
    base_url : str
        Source directory URL.
    categories : list[str]
        Categories to download (details, fatalities, locations).
    start_year : int
        Start year (inclusive).
    end_year : int
        End year (inclusive).
    output_dir : str
        Output directory for Parquet files.
    """
    session = requests.Session()
    print(f"Listing files from {base_url} ...")
    files = list_available_files(session, base_url)
    print(f"Found {len(files)} files in directory listing.")

    filtered = filter_files(files, categories, start_year, end_year)
    print(f"Filtered down to {len(filtered)} files for categories {categories} "
          f"from {start_year} to {end_year}.")

    # Organize files by category
    files_by_cat = defaultdict(list)
    for cat, year, fname in filtered:
        files_by_cat[cat].append((year, fname))

    for cat, year_files in files_by_cat.items():
        print(f"Processing category '{cat}' with {len(year_files)} files.")
        dfs = []
        for year, fname in sorted(year_files):
            print(f" Downloading {fname} for year {year} ...")
            df = download_and_parse(session, base_url, fname)
            df['year'] = year  # Optionally annotate data with year
            dfs.append(df)

        if dfs:
            save_parquet(dfs, cat, start_year, end_year, output_dir=output_dir)
        else:
            print(f"No data frames to process for category '{cat}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and aggregate NOAA Storm Events (SED) CSVs into per-category Parquet files.")
    parser.add_argument("--start-year", type=int, required=True, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, required=True, help="End year (inclusive)")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, help="Categories to download; default: all")
    parser.add_argument("--base-url", default=BASE_URL_DEFAULT, help="Storm Events CSV directory URL")
    parser.add_argument("--out-dir", default="data", help="Output directory (default: data)")

    args = parser.parse_args()
    cats = args.categories if args.categories else CATEGORIES

    main(
        base_url=args.base_url,
        categories=cats,
        start_year=args.start_year,
        end_year=args.end_year,
        output_dir=args.out_dir,
    )