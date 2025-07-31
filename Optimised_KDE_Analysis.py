"""
This script processes historical climate data and compares it with ERA5 weather reanalysis data. It includes functionalities to:

1. Read and process PSV data files, extracting and filtering observations based on temperature and pressure ranges.
2. Load data in parallel for efficiency and assign location IDs based on first appearances.
3. Extract and correlate ERA5 data with historical records, matching them by coordinates and time range.
4. Plot KDEs for comparisons, computing metrics (MSE, RMSE and KL Divergence) to assess fit.
5. Comparison plots and metrics are saved to the output directory.
6. Comparison metrics are exported as CSV files.

Ensure directories for historical data, grib files and output are correctly specified before running the script. Set the minimum number of observation days needed. 

"""

import os
import re
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde
from math import radians, sin, cos, sqrt, atan2
from io import StringIO
from concurrent.futures import ProcessPoolExecutor

# === Directories ===

# Set data directories
historical_folder = r" "

grib_folder = r" "

output_folder = r" "

os.makedirs(output_folder, exist_ok=True)

# Set the minimum number of observation days
MIN_DAYS = 100

# === Helper Functions ===
def read_psv(path):
    header, rows = None, []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Source_ID"):
                if header is None:
                    header = line
            else:
                rows.append(line)
    if header is None:
        raise ValueError(f"No 'Source_ID' header in {path}")
    return pd.read_csv(StringIO("\n".join([header] + rows)),
                       sep="|", low_memory=False)

def get_station_id_from_psv(fname):
    parts = fname.split("_")
    return parts[0]

def get_variable_from_filename(fname):
    n = fname.lower()
    if "temperature" in n:
        return "temperature"
    if "pressure" in n:
        return "pressure"
    return "unknown"

def process_psv_file(fname):
    try:
        df = read_psv(os.path.join(historical_folder, fname))
        station = get_station_id_from_psv(fname)
        variable = get_variable_from_filename(fname)

        df["station_id_file"] = station
        df["station_id_col"] = df.get("Station_ID", station).astype(str)
        df["variable"] = variable
        df["datetime"] = pd.to_datetime(
            dict(year=df["Year"], month=df["Month"],
                 day=df["Day"], hour=df["Hour"]))
        if "Latitude" not in df or "Longitude" not in df:
            raise ValueError("latitude / longitude missing")
        df["lat"] = df["Latitude"]
        df["lon"] = df["Longitude"]

        if variable == "pressure":
            df = df[(df["Observed_value"] >= 900) &
                    (df["Observed_value"] <= 1050)]

        if variable == "temperature":
            df = df[(df["Observed_value"] >= -90) &
                    (df["Observed_value"] <= 60)]

        return df[["station_id_file", "station_id_col", "variable",
                   "datetime", "lat", "lon", "Observed_value",
                   "Elevation"]]
    except Exception as e:
        print(f"Failed to read {fname}: {e}")
        return None

def load_all_historical_parallel(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".psv")]
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_psv_file, files)
    return pd.concat([r for r in results if r is not None], ignore_index=True)


def extract_era5_timeseries(station_id, variable, req_lat, req_lon, start, end):
    var2filename = {"temperature": "2m_temperature", "pressure": "surface_pressure"}
    var2key = {"temperature": "t2m", "pressure": "sp"}
    var_name = var2filename[variable]
    data_key = var2key[variable]

    file_name = f"{station_id}_{var_name}.nc"
    file_path = os.path.join(grib_folder, file_name)

    if not os.path.exists(file_path):
        print(f"ERA5 file not found: {file_path}")
        return None, None, None

    ds = xr.open_dataset(file_path, engine="netcdf4")

    print(f"  Opened ERA5 file: {file_path}")
    print(f"  ERA5 dataset dimensions: {ds.sizes}")

    # Select the closest grid point to req_lat, req_lon
    lat_diff = np.abs(ds['latitude'].values - req_lat)
    lon_diff = np.abs(ds['longitude'].values - req_lon)
    idx_lat = np.argmin(lat_diff)
    idx_lon = np.argmin(lon_diff)
    closest_lat = ds['latitude'].values[idx_lat]
    closest_lon = ds['longitude'].values[idx_lon]

    var_da = ds[data_key]
    
    times = pd.to_datetime(var_da.isel(latitude=idx_lat, longitude=idx_lon).time.values)
    data = var_da.isel(latitude=idx_lat, longitude=idx_lon).values
    mask = (times >= start) & (times <= end)
    df = pd.DataFrame({'datetime': times[mask], 'era5_value': data[mask]})
    if len(df) == 0:
        print(f"ERA5 data insufficient for station={station_id}, var={variable}, lat={req_lat}, lon={req_lon}")
        return None, None, None


    if variable == "temperature":
        df['era5_value'] = df['era5_value'] - 273.15
    else:
        df['era5_value'] = df['era5_value'] / 100.0

    return df, closest_lat, closest_lon

def assign_location_first_appearance(df):
    df["latlon_str"] = df["lat"].astype(str) + "_" + df["lon"].astype(str)
    df["location_first_loc_id"] = (
        df.groupby(["station_id_file", "variable", "latlon_str"]).ngroup() + 1
    )
    return df.drop(columns="latlon_str")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# === Plotting Routines ===
def plot_station_variable_pdfs_with_era5(hist_df, station_id, variable, locations_df):
    # Sort the locations by start date
    locations_df = locations_df.sort_values(by='start_dt')

    # Find the earliest valid location
    earliest_valid_loc = locations_df.iloc[0]

    # Compute the reference KDE (earliest valid location)
    lat, lon = earliest_valid_loc['lat'], earliest_valid_loc['lon']
    start, end = earliest_valid_loc['start_dt'], earliest_valid_loc['end_dt']
    val_subset = hist_df[
        (hist_df['station_id_file'] == station_id) &
        (hist_df['variable'] == variable) &
        (hist_df['lat'] == lat) &
        (hist_df['lon'] == lon)
    ]['Observed_value'].dropna()
    kde_ref = gaussian_kde(val_subset)
    x_grid = np.linspace(val_subset.min(), val_subset.max(), 100)
    ref_kde_values = kde_ref(x_grid)

    # Compare the other KDEs to the reference KDE
    mse_values = []
    kl_divergence_values = []
    metric_rows = []

    for idx, loc in locations_df.iterrows():
        lat, lon = loc['lat'], loc['lon']
        val_subset = hist_df[
            (hist_df['station_id_file'] == station_id) &
            (hist_df['variable'] == variable) &
            (hist_df['lat'] == lat) &
            (hist_df['lon'] == lon)
        ]['Observed_value'].dropna()
        kde = gaussian_kde(val_subset)
        kde_values = kde(x_grid)
        kl_divergence_to_ref = np.sum(ref_kde_values * np.log(ref_kde_values / kde_values + 1e-12))
        kl_divergence_reverse = np.sum(kde_values * np.log(kde_values / ref_kde_values + 1e-12))
        symmetric_kl_divergence = (kl_divergence_to_ref + kl_divergence_reverse) / 2
        mse = np.mean((kde_values - ref_kde_values) ** 2)
        mse_values.append(mse)
        kl_divergence_values.append(symmetric_kl_divergence)
        # Gather info for export
        metric_rows.append({
            "station_id": station_id,
            "variable": variable,
            "location_index": idx + 1,
            "lat": lat,
            "lon": lon,
            "start_dt": loc['start_dt'],
            "end_dt": loc['end_dt'],
            "mse": mse,
            "kl_divergence": symmetric_kl_divergence,
        })

    mse_values = np.array(mse_values)
    kl_divergence_values = np.array(kl_divergence_values)
    rmse_values = np.sqrt(mse_values)
    
    # Add RMSE to export
    for i in range(len(metric_rows)):
        metric_rows[i]["rmse"] = rmse_values[i]

    # Plot the KDEs
    plt.figure(figsize=(10, 8))
    color_iter = iter(plt.cm.tab20.colors)
    for _, loc in locations_df.iterrows():
        lat, lon = loc['lat'], loc['lon']
        val_subset = hist_df[
            (hist_df['station_id_file'] == station_id) &
            (hist_df['variable'] == variable) &
            (hist_df['lat'] == lat) &
            (hist_df['lon'] == lon)
        ]['Observed_value'].dropna()
        kde = gaussian_kde(val_subset)
        c = next(color_iter, None)
        plt.plot(x_grid, kde(x_grid), label=f"({lat:.3f},{lon:.3f})", color=c, lw=2)
    
    era5_result = extract_era5_timeseries(station_id, variable, earliest_valid_loc['lat'], earliest_valid_loc['lon'], earliest_valid_loc['start_dt'], earliest_valid_loc['end_dt'])
    era5_df, era5_lat, era5_lon = era5_result
    if era5_df is not None and not era5_df.empty:
        era5_vals = era5_df['era5_value'].dropna()
        if len(era5_vals) >= 10:
            kde_era5 = gaussian_kde(era5_vals)  # Note the assignment
            plt.plot(x_grid, kde_era5(x_grid), label=f"ERA5 (Ref) ({era5_lat:.3f},{era5_lon:.3f})", color='black', linestyle='-.', lw=2)
    plt.plot(x_grid, ref_kde_values, label=f"Ref ({earliest_valid_loc['lat']:.3f},{earliest_valid_loc['lon']:.3f})", color='black', lw=2, linestyle='--')
    plt.title(f"KDE Comparison for Station: {station_id}, Variable: {variable} (Min No of Obs: {MIN_DAYS} days)")
    plt.xlabel(f"{variable.capitalize()} Value")
    plt.ylabel("Density")
    plt.legend(fontsize=9, loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f"allloc_pdf_{station_id}_{variable}.png"), dpi=150)
    plt.close()

    # Create a 3-panel plot for MSE, RMSE, and KL Divergence
    fig, axs = plt.subplots(3, figsize=(11, 10))
    axs[0].plot(mse_values, marker='o')
    axs[0].set_title("MSE between KDEs and reference KDE")
    axs[0].set_xlabel("Location Index")
    axs[0].set_ylabel("MSE")
    axs[0].set_xticks(range(len(mse_values)))
    axs[0].set_xticklabels([f"{lat:.3f},{lon:.3f}" for lat, lon in zip(locations_df['lat'], locations_df['lon'])], rotation=90)

    axs[1].plot(rmse_values, marker='o', color='orange')
    axs[1].set_title("RMSE between KDEs and reference KDE")
    axs[1].set_xlabel("Location Index")
    axs[1].set_ylabel("RMSE")
    axs[1].set_xticks(range(len(rmse_values)))
    axs[1].set_xticklabels([f"{lat:.3f},{lon:.3f}" for lat, lon in zip(locations_df['lat'], locations_df['lon'])], rotation=90)

    axs[2].plot(kl_divergence_values, marker='o', color='green')
    axs[2].set_title("Symmetric KL Divergence between KDEs and reference KDE")
    axs[2].set_xlabel("Location Index")
    axs[2].set_ylabel("KL Divergence")
    axs[2].set_xticks(range(len(kl_divergence_values)))
    axs[2].set_xticklabels([f"{lat:.3f},{lon:.3f}" for lat, lon in zip(locations_df['lat'], locations_df['lon'])], rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"mse_rmse_kl_divergence_{station_id}_{variable}.png"), dpi=150)
    plt.close()

    # Export to CSV
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(output_folder, f"comparison_metrics_{station_id}_{variable}.csv"),
        index=False)

    print(f"Exported metrics CSV: {os.path.join(output_folder, f'comparison_metrics_{station_id}_{variable}.csv')}")

def plot_ob_minus_background(hist_df, station_id, variable, MIN_DAYS):
    subset_all = hist_df[(hist_df["station_id_file"] == station_id) & (hist_df["variable"] == variable)].copy()
    if subset_all.empty:
        print(f"No data for station {station_id} and variable {variable}.")
        return

    subset_all["date"] = subset_all["datetime"].dt.date
    stats = (subset_all.groupby(["lat", "lon", "location_first_loc_id"])
                         .agg(uniq_days=("date", "nunique"),
                              start_dt=("datetime", "min"),
                              end_dt=("datetime", "max"))
                         .reset_index())
    stats = stats[stats["uniq_days"] >= MIN_DAYS]
    if stats.empty:
        print(f"No valid groups with {MIN_DAYS} days for station {station_id} and variable {variable}.")
        return

    plt.figure(figsize=(10, 7))
    colour_map = plt.cm.tab10

    for _, row in stats.sort_values("location_first_loc_id").iterrows():
        loc_id, lat, lon = int(row["location_first_loc_id"]), row["lat"], row["lon"]
        sub = subset_all[(subset_all["lat"] == lat) & (subset_all["lon"] == lon)]

        df_era5, era5_lat, era5_lon = extract_era5_timeseries(
            station_id, variable, lat, lon, row["start_dt"], row["end_dt"])
        if df_era5 is None or df_era5.empty:
            print(f"ERA5 data missing for loc_id {loc_id}.")
            continue

        obs_df = sub[["datetime", "Observed_value"]].copy()
        obs_df["datetime"] = pd.to_datetime(obs_df["datetime"]).dt.floor("h")
        obs_df = obs_df.groupby("datetime", as_index=False).mean()

        era5_df = df_era5[["datetime", "era5_value"]].copy()
        era5_df["datetime"] = pd.to_datetime(era5_df["datetime"]).dt.floor("h")
        era5_df = era5_df.groupby("datetime", as_index=False).mean()

        merged = pd.merge(obs_df, era5_df, on="datetime", how="inner")
        merged = merged.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  Merged dataframe has {len(merged)} rows")
        
        if len(merged) < 5:
            print(f"  Skipping plot due to insufficient data (n={len(merged)})")
            continue

        diff = merged["Observed_value"] - merged["era5_value"]
        print(f"Plotting {len(diff)} differences for loc_id {loc_id}.")

        kde = gaussian_kde(diff)
        xgrid = np.linspace(diff.min(), diff.max(), 200)
        colour = colour_map((loc_id - 1) % 10)

        plt.plot(xgrid, kde(xgrid), lw=2, color=colour,
                 label=(f"loc_id{loc_id} "
                        f"({lat:.3f},{lon:.3f}) "
                        f"n={len(diff)}"))

    plt.title(f"O − B Gaussian KDE  :  {station_id}   {variable} (Min No of Obs: {MIN_DAYS} days)")
    plt.xlabel(f"{variable.capitalize()}  (Observation − Background)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.savefig(os.path.join(output_folder, f"pdf_OB_{station_id}_{variable}.png".replace(' ', '_')), dpi=150)
    plt.close()

# === Main Process ===
def main():
    hist_df_path = os.path.join(output_folder, "hist_df.pkl")
    valid_groups_path = os.path.join(output_folder, "valid_groups.pkl")
    era5_rows_path = os.path.join(output_folder, "era5_rows.pkl")

    # Check if historical data is already loaded
    if os.path.exists(hist_df_path):
        print("Loading historical data from checkpoint...")
        with open(hist_df_path, "rb") as f:
            hist_df = pickle.load(f)
    else:
        print("Loading historical PSV data …")
        hist_df = load_all_historical_parallel(historical_folder)
        print(f"Loaded {len(hist_df)} records")
        # Save historical data to checkpoint file
        with open(hist_df_path, "wb") as f:
            pickle.dump(hist_df, f)

    # Check if valid groups are already computed
    if os.path.exists(valid_groups_path):
        print("Loading valid groups from checkpoint...")
        with open(valid_groups_path, "rb") as f:
            valid = pickle.load(f)
        hist_df = assign_location_first_appearance(hist_df)
    else:
        print("Assigning location ids …")
        hist_df = assign_location_first_appearance(hist_df)

        hist_df['date'] = hist_df['datetime'].dt.date
        group_cols = ['station_id_file', 'variable', 'lat', 'lon']
        group_stats = (
            hist_df.groupby(group_cols)
            .agg(unique_days=('date', 'nunique'), start_dt=('datetime', 'min'), end_dt=('datetime', 'max'))
            .reset_index()
        )
        valid = group_stats[group_stats['unique_days'] >= MIN_DAYS]
        print(f"Found {len(valid)} station-variable-location groups.")
        # Save valid groups to checkpoint file
        with open(valid_groups_path, "wb") as f:
            pickle.dump(valid, f)

    # Plotting routines
    for _, key in hist_df[["station_id_file", "variable"]].drop_duplicates().iterrows():
        plot_ob_minus_background(hist_df,
                                 key["station_id_file"],
                                 key["variable"],
                                 MIN_DAYS)

    station_var_keys = valid[['station_id_file', 'variable']].drop_duplicates()

    for _, key in station_var_keys.iterrows():
        station_id, variable = key['station_id_file'], key['variable']
        locs = valid[
            (valid['station_id_file'] == station_id) & (valid['variable'] == variable)
        ]
        print(f"  Station {station_id}, variable {variable}: {len(locs)} locations")
        if len(locs) == 0:
            continue
        plot_station_variable_pdfs_with_era5(hist_df, station_id, variable, locs)

    # ERA5 processing
    if os.path.exists(era5_rows_path):
        print("Loading ERA5 rows from checkpoint...")
        with open(era5_rows_path, "rb") as f:
            era5_rows = pickle.load(f)
    else:
        era5_rows = []
        for _, row in valid.iterrows():
            station_id = row['station_id_file']
            variable = row['variable']
            lat = row['lat']
            lon = row['lon']
            start, end = row['start_dt'], row['end_dt']

            location_id = hist_df[
                (hist_df['station_id_file'] == station_id) &
                (hist_df['variable'] == variable) &
                (hist_df['lat'] == lat) &
                (hist_df['lon'] == lon)
            ]['location_first_loc_id'].iloc[0]

            first_loc = hist_df[
                (hist_df['station_id_file'] == station_id) &
                (hist_df['variable'] == variable)
            ].sort_values('datetime').iloc[0]
            first_lat = first_loc['lat']
            first_lon = first_loc['lon']

            mean_elevation = hist_df[
                (hist_df['station_id_file'] == station_id) &
                (hist_df['variable'] == variable)
            ]['Elevation'].mean()
            
            
            era5_result = extract_era5_timeseries(station_id, variable, lat, lon, start, end)
            if era5_result and era5_result[0] is not None:  # Check if era5_df is not None
                era5_df, era5_lat, era5_lon = era5_result
                if not era5_df.empty:
                    distance_to_first_loc = haversine(first_lat, first_lon, lat, lon)
                    distance_to_era5 = haversine(lat, lon, era5_lat, era5_lon)
                else:
                    era5_lat = np.nan
                    era5_lon = np.nan
                    distance_to_first_loc = np.nan
                    distance_to_era5 = np.nan
            else:
                era5_df = None  # Set era5_df to None if era5_result is None or era5_result[0] is None
                era5_lat = np.nan
                era5_lon = np.nan
                distance_to_first_loc = np.nan
                distance_to_era5 = np.nan
            
            
            era5_rows.append({
                'station_id': station_id,
                'variable': variable,
                'location_id': location_id,
                'hist_lat': lat,
                'hist_lon': lon,
                'first_lat': first_lat,
                'first_lon': first_lon,
                'era5_lat': era5_lat,
                'era5_lon': era5_lon,
                'distance_to_first_loc_km': distance_to_first_loc,
                'distance_to_era5_km': distance_to_era5,
                'mean_elevation_m': mean_elevation
            })
    # Save ERA5 rows to checkpoint file
    with open(era5_rows_path, "wb") as f:
        pickle.dump(era5_rows, f)

    out_csv_path = os.path.join(output_folder, "era5_latlon_distance_lookup.csv")
    pd.DataFrame(era5_rows).to_csv(out_csv_path, index=False)
    print("Exported ERA5 location comparison CSV:", out_csv_path)

    print("All done! Results saved in:", output_folder)

if __name__ == "__main__":
    main()