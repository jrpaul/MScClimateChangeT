# Identifying Undocumented Station Relocations using KDE

This script processes historical climate data and compares it with ERA5 weather reanalysis data. It includes functionalities to:

1. Read and process PSV data files, extracting and filtering observations based on temperature and pressure ranges.
2. Load data in parallel for efficiency and assign location IDs based on first appearances.
3. Extract and correlate ERA5 data with historical records, matching them by coordinates and time range.
4. Plot KDEs for comparisons, computing metrics (MSE, RMSE and KL Divergence) to assess fit.
5. Comparison plots and metrics are saved to the output directory.
6. Comparison metrics are exported as CSV files.

Ensure directories for historical data, grib files and output are correctly specified before running the script. Set the minimum number of observation days needed. 
