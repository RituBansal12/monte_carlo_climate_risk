# monte_carlo_climate_risk
Climate Risk to Supply Chains with Monte Carlo Simulations

## Data Schema

Schemas detected under `data/`.

### epiNOAA_cty_scaled_1950_2025_ALL.parquet
- region_type: string
- fips: string
- ncei_code: string
- state_name: string
- postal_code: string
- region_name: string
- date: timestamp[ns]
- tmax: string
- tmin: string
- tavg: string
- prcp: string
- YEAR: int32
- STATUS: string

Notes:
- This file is an EpiNOAA county-level parquet compiled via PyArrow from NOAA S3 and includes Hive partition columns `YEAR` and `STATUS`.
- Some climate metrics (tmax, tmin, tavg, prcp) are currently typed as strings in the parquet and may require casting to numeric for analysis.

### events-US-1980-2024-Q4.csv
- Name: object
- Disaster: object
- Begin Date: int64
- End Date: int64
- CPI-Adjusted Cost($M): float64
- Unadjusted Cost($M): float64
- Deaths: int64

Notes:
- Types are inferred via pandas on read and may change if the file content changes.

### SED_details_1950-2025.parquet
- begin_yearmonth: string
- begin_day: string
- begin_time: string
- end_yearmonth: string
- end_day: string
- end_time: string
- episode_id: string
- event_id: string
- state: string
- state_fips: string
- year: int32
- month_name: string
- event_type: string
- cz_type: string
- cz_fips: string
- cz_name: string
- wfo: string
- begin_date_time: string
- cz_timezone: string
- end_date_time: string
- injuries_direct: string
- injuries_indirect: string
- deaths_direct: string
- deaths_indirect: string
- damage_property: string
- damage_crops: string
- source: string
- magnitude: string
- magnitude_type: string
- flood_cause: string
- category: string
- tor_f_scale: string
- tor_length: string
- tor_width: string
- tor_other_wfo: string
- tor_other_cz_state: string
- tor_other_cz_fips: string
- tor_other_cz_name: string
- begin_range: string
- begin_azimuth: string
- begin_location: string
- end_range: string
- end_azimuth: string
- end_location: string
- begin_lat: string
- begin_lon: string
- end_lat: string
- end_lon: string
- episode_narrative: string
- event_narrative: string

Notes:
- Columns follow NOAA Storm Events Details. Parsed as strings; cast to numeric as needed. `year` is added by `02_download_SED.py`.

### SED_locations_1950-2025.parquet
- episode_id: string
- event_id: string
- location_index: string
- range: string
- azimuth: string
- location: string
- lat: string
- lon: string
- year: int32

Notes:
- NOAA Storm Events Locations schema. Parsed as strings; `year` added by `02_download_SED.py`.

### SED_fatalities_1950-2025.parquet
- fatality_id: string
- event_id: string
- fatality_type: string
- fatality_date: string
- fatality_age: string
- fatality_sex: string
- fatality_location: string
- year: int32

Notes:
- NOAA Storm Events Fatalities schema. Parsed as strings; `year` added by `02_download_SED.py`.

## Data Citations
Smith, Adam B. (2020). U.S. Billion-dollar Weather and Climate Disasters, 1980 - present (NCEI Accession 0209268). NOAA National Centers for Environmental Information. Dataset. https://doi.org/10.25921/stkw-7w73. Accessed 2025-08-23.

NOAA U.S. Climate Gridded Dataset (NClimGrid) was accessed on 2025-08-23 from https://registry.opendata.aws/noaa-nclimgrid.

National Centers for Environmental Information. Storm Events Database. NOAA National Centers for Environmental Information. Dataset. https://www.ncei.noaa.gov/stormevents/. Accessed 2025-08-23.