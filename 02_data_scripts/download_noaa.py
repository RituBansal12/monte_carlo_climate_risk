import os
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import fs as pafs
import argparse

def parse_years_args(tokens):
    """Parse years from a list of tokens that may include ranges like '2019-2021'.

    Returns a sorted list of unique integer years.
    """
    years = set()
    for tok in tokens:
        tok = str(tok)
        if "-" in tok:
            start_s, end_s = tok.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            years.update(range(start, end + 1))
        else:
            years.add(int(tok))
    if not years:
        raise ValueError("No valid years provided.")
    return sorted(years)

def load_epiNOAA_data(years, months=None, status="scaled", spatial_res: str = "ste"):
    """
    Load EpiNOAA daily climate data from NOAA S3 (state 'ste' or county 'cty').

    Parameters
    ----------
    years : list[int]
        List of years to download (e.g., [2000, 2001]).
    months : list[int], optional
        List of months (1–12). If None, all months are included.
    status : str
        Data status: 'scaled' or 'raw'.
    spatial_res : str
        Spatial resolution: 'ste' (state) or 'cty' (county).

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all requested data.
    """
    # Read from the dataset ROOT and let PyArrow discover Hive-style partitions.
    # When passing an S3FileSystem instance, pyarrow expects 'bucket/key' paths (no 's3://').
    if spatial_res not in {"ste", "cty"}:
        raise ValueError("spatial_res must be one of {'ste','cty'}")
    base = f"noaa-nclimgrid-daily-pds/EpiNOAA/v1-0-0/parquet/{spatial_res}/"
    print(f"Opening dataset at s3://{base} (anonymous S3)...")

    s3 = pafs.S3FileSystem(anonymous=True)
    dataset = ds.dataset(base, filesystem=s3, format="parquet", partitioning="hive")

    # Build partition filters where fields exist: YEAR (as strings), optional MONTH (zero-padded), and STATUS
    schema_fields = set(dataset.schema.names)
    filt = None
    # YEAR is expected
    year_filter = ds.field("YEAR").isin([str(y) for y in years]) if "YEAR" in schema_fields else None
    status_filter = (ds.field("STATUS") == status) if "STATUS" in schema_fields else None
    month_filter = None
    if months and "MONTH" in schema_fields:
        month_vals = [f"{m:02d}" for m in months]
        month_filter = ds.field("MONTH").isin(month_vals)

    # Combine filters
    for part in [year_filter, status_filter, month_filter]:
        if part is None:
            continue
        filt = part if filt is None else (filt & part)

    print("Applying filters:", {
        "YEAR": years,
        "MONTH": months if months else "ALL",
        "STATUS": status,
    })

    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        raise ValueError("No data loaded. Check years/months/status inputs.")

    return table.to_pandas()


def save_parquet(df, output_path, filename_base):
    """
    Save DataFrame to Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : str
        Output directory (e.g., 'data/').
    filename_base : str
        Base filename without extension.
    """
    os.makedirs(output_path, exist_ok=True)
    parquet_path = os.path.join(output_path, f"{filename_base}.parquet")
    print(f"Saving Parquet to {parquet_path}")
    df.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EpiNOAA parquet from NOAA S3")
    parser.add_argument("--years", type=str, nargs="+", required=True,
                        help="Years or ranges (inclusive), e.g. --years 2019 2020-2021")
    parser.add_argument("--months", type=int, nargs="+",
                        help="Months 1-12; omit to include all months")
    parser.add_argument("--status", choices=["scaled", "raw"], default="scaled",
                        help="Data status to load")
    parser.add_argument("--spatial-res", choices=["ste", "cty"], default="ste",
                        help="Spatial resolution: ste (state) or cty (county)")
    parser.add_argument("--out-dir", default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--output-name", default=None,
                        help="Base filename without extension; auto-generated if omitted")

    args = parser.parse_args()

    years_list = parse_years_args(args.years)
    df = load_epiNOAA_data(years_list, months=args.months, status=args.status, spatial_res=args.spatial_res)

    # Determine output filename
    if args.output_name:
        base_name = args.output_name
    else:
        yrs = years_list
        yrs_part = f"{min(yrs)}_{max(yrs)}" if len(yrs) > 1 else f"{yrs[0]}"
        mon_part = "ALL" if not args.months else "M" + "-".join(f"{m:02d}" for m in args.months)
        base_name = f"epiNOAA_{args.spatial_res}_{args.status}_{yrs_part}_{mon_part}"

    print(f"Loaded {df.shape[0]:,} rows. Preparing to save...")
    save_parquet(df, args.out_dir, base_name)
    print(f"Done. Wrote: {os.path.join(args.out_dir, base_name + '.parquet')}" )
