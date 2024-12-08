import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
import os
from datetime import datetime
import glob
from tqdm import tqdm
from rasterio.enums import Resampling
from multiprocessing import Pool
from itertools import product
import rioxarray
import time
import re

# Define config and functions
data_dir = r"/home/waves/projects/smap-drydown/data"
SMAPL3_dir = "SPL3SMP"
datarods_dir = "datarods"
SMAPL4_dir = "SPL4SMGP"
SMAPL4_grid_dir = "SMAPL4SMGP_EASEreference"
rangeland_dir = "rap-vegetation-cover-v3"


def create_output_dir(out_dir):
    # Create and save the datarods
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"created {out_dir}")
    return out_dir


def get_filepath_from_pattern(filename_pattern, directory):
    file_paths = glob.glob(os.path.join(directory, filename_pattern))
    print(f"{filename_pattern}: {len(file_paths)} files available")
    return file_paths


#################################################
# Get EASEgrid template from SMAP
#################################################


class EASEgrid_template:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.row_index, self.column_index, self.longitude, self.latitude = (
            self.get_grid_coordinate()
        )
        self.varname = "SPL3SMP"
        self.timestep = 365
        self.filenames = self.get_filepath()
        self.data = self.get_template_dataset()

    def get_grid_coordinate(self):
        SMAPL3_grid_sample = os.path.join(
            self.data_dir, r"SPL3SMP/SMAP_L3_SM_P_20150331_R18290_001.h5"
        )

        ncf = netCDF4.Dataset(SMAPL3_grid_sample, diskless=True, persist=False)
        nch_am = ncf.groups.get("Soil_Moisture_Retrieval_Data_AM")
        nch_pm = ncf.groups.get("Soil_Moisture_Retrieval_Data_PM")

        # Return as regular numpy array rather than masked array
        _latitude = ma.getdata(
            nch_am.variables["latitude"][:].filled(fill_value=np.nan), subok=True
        )
        _longitude = ma.getdata(
            nch_am.variables["longitude"][:].filled(fill_value=np.nan), subok=True
        )
        _EASE_column_index = ma.getdata(
            nch_am.variables["EASE_column_index"][:].astype(int).filled(fill_value=-1),
            subok=True,
        )
        _EASE_row_index = ma.getdata(
            nch_am.variables["EASE_row_index"][:].astype(int).filled(fill_value=-1),
            subok=True,
        )

        # Coordinates with no data are skipped --- fill them
        latitude = np.nanmax(_latitude, axis=1)
        EASE_row_index = np.nanmax(_EASE_row_index, axis=1)
        longitude = np.nanmax(_longitude, axis=0)
        EASE_column_index = np.nanmax(_EASE_column_index, axis=0)

        return EASE_row_index, EASE_column_index, longitude, latitude

    def get_filepath(self):
        filepaths = get_filepath_from_pattern(
            filename_pattern=f"SMAP_L3_SM_P_*.h5",
            directory=f"{data_dir}/{self.varname}",
        )
        return filepaths

    def get_template_dataset(self):
        _ds_SMAPL3 = xr.open_dataset(
            self.filenames[0],
            engine="rasterio",
            group="Soil_Moisture_Retrieval_Data_AM",
            variable=["soil_moisture"],
        )
        ds_SMAPL3_coord_template = _ds_SMAPL3.assign_coords(
            {"x": self.longitude, "y": self.latitude}
        ).rio.write_crs("epsg:4326")
        return ds_SMAPL3_coord_template


def main():

    print("Resample rangeland data to SMAP L3 EASE grids")

    # Execute
    SMAPL3_grid_sample = os.path.join(
        data_dir, r"SPL3SMP/SMAP_L3_SM_P_20150331_R18290_001.h5"
    )
    ncf = netCDF4.Dataset(SMAPL3_grid_sample, diskless=True, persist=False)
    nch_am = ncf.groups.get("Soil_Moisture_Retrieval_Data_AM")
    nch_pm = ncf.groups.get("Soil_Moisture_Retrieval_Data_PM")

    # Return as regular numpy array rather than masked array
    _latitude = ma.getdata(
        nch_am.variables["latitude"][:].filled(fill_value=np.nan), subok=True
    )
    _longitude = ma.getdata(
        nch_am.variables["longitude"][:].filled(fill_value=np.nan), subok=True
    )
    _EASE_column_index = ma.getdata(
        nch_am.variables["EASE_column_index"][:].astype(int).filled(fill_value=-1),
        subok=True,
    )
    _EASE_row_index = ma.getdata(
        nch_am.variables["EASE_row_index"][:].astype(int).filled(fill_value=-1),
        subok=True,
    )

    # Coordinates with no data are skipped --- fill them
    latitude = np.nanmax(_latitude, axis=1)
    EASE_row_index = np.nanmax(_EASE_row_index, axis=1)
    longitude = np.nanmax(_longitude, axis=0)
    EASE_column_index = np.nanmax(_EASE_column_index, axis=0)

    filepaths = get_filepath_from_pattern(
        filename_pattern=f"SMAP_L3_SM_P_*.h5", directory=f"{data_dir}/SPL3SMP"
    )
    _ds_SMAPL3 = xr.open_dataset(
        filepaths[0],
        engine="rasterio",
        group="Soil_Moisture_Retrieval_Data_AM",
        variable=["soil_moisture"],
    )
    ds_SMAPL3_coord_template = _ds_SMAPL3.assign_coords(
        {"x": longitude, "y": latitude}
    ).rio.write_crs("epsg:4326")
    ease_template = (
        ds_SMAPL3_coord_template.Soil_Moisture_Retrieval_Data_AM_soil_moisture
    )

    #################################################
    # Get rangeland data
    #################################################

    # Configs
    out_dir = create_output_dir(os.path.join(data_dir, "rangeland_resampled_avg"))

    # Get original files
    filenames = get_filepath_from_pattern(
        filename_pattern=f"vegetation-cover-v3-*.tif",
        directory=os.path.join(data_dir, "rap-vegetation-cover-v3"),
    )
    print("Files found: ")
    print(filenames)

    for i, filename in enumerate(filenames):
        # Open datasets
        _ds = rioxarray.open_rasterio(filename)
        ds = _ds.rio.write_crs("epsg:4326", inplace=True)

        # Subset according to bbox
        left, bottom, right, top = ds.rio.bounds()
        subset_ease_template = ease_template.sel(
            x=slice(left, right), y=slice(top, bottom)
        )

        # Using regular expression to find year pattern in the string
        match = re.search(r"\d{4}", filename)
        record_year = match.group() if match else "Year not found"

        # Loop through bands
        for band_num in range(1, 7):  # This will loop from 1 to 6
            # Select one vegetation type and start resample
            veg_ds = ds.sel(band=band_num)

            # print(
            #     f"Currently masking the data of Year {record_year} - band {band_num}"
            # )
            # start_time = time.time()
            # veg_ds = veg_ds.where(veg_ds != veg_ds._FillValue, np.nan)
            # end_time = time.time()
            # elapsed_time = end_time - start_time
            # print(
            #     f"Finished masking the data of Year {record_year} - band {band_num}\nTime taken for the operation: {elapsed_time} seconds"
            # )

            print(
                f"Currently resampling the data of Year {record_year} - band {band_num}"
            )
            start_time = time.time()
            # veg_ds_resampled = veg_ds.rio.reproject_match(subset_ease_template, resampling=Resampling.average) # Somehow this didn't align with the coastline
            # Example of defining new grid boundaries manually or from subset_ease_template

            # grid_interval_y_min = abs(
            #     subset_ease_template.y.data[0] - subset_ease_template.y.data[1]
            # )
            # grid_interval_x_min = abs(
            #     subset_ease_template.x.data[0] - subset_ease_template.x.data[1]
            # )
            # # For maximum edge using the last two elements
            # grid_interval_y_max = abs(
            #     subset_ease_template.y.data[-1] - subset_ease_template.y.data[-2]
            # )
            # grid_interval_x_max = abs(
            #     subset_ease_template.x.data[-1] - subset_ease_template.x.data[-2]
            # )

            new_lat_bounds = np.linspace(
                start=min(subset_ease_template.y),  # - grid_interval_y_min,
                stop=max(subset_ease_template.y),  # + grid_interval_y_max,
                num=len(subset_ease_template.y) + 1,
            )
            new_lon_bounds = np.linspace(
                start=min(subset_ease_template.x),  # - grid_interval_x_min,
                stop=max(subset_ease_template.x),  # + grid_interval_x_max,
                num=len(subset_ease_template.x) + 1,
            )

            # Grouping by these new bins
            _veg_ds_resampled = veg_ds.groupby_bins("y", new_lat_bounds).mean()
            veg_ds_resampled = _veg_ds_resampled.groupby_bins(
                "x", new_lon_bounds
            ).mean()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Finished resampling the data of Year {record_year} - band {band_num}\nTime taken for the operation: {elapsed_time} seconds"
            )

            out_ds = xr.DataArray(
                data=veg_ds_resampled.data,
                dims=["y", "x"],
                coords={
                    "y": np.flip(subset_ease_template.y.data),
                    "x": subset_ease_template.x.data,
                },
            )
            out_ds.attrs["crs"] = "EPSG:4326"

            out_filepath = os.path.join(out_dir, f"{record_year}_band{band_num}.nc")
            out_ds.to_netcdf(path=out_filepath)


if __name__ == "__main__":
    main()

# # Plotting

# import cartopy.crs as ccrs

# projection = ccrs.PlateCarree()
# fig, ax = plt.subplots(subplot_kw={"projection": projection})
# conus_extent = [-125, -66.5, 24.5, 49.5]
# # Plot the new data
# veg_ds_resampled.plot(ax=ax, transform=projection, vmin=0, vmax=1)
# ax.set_extent(conus_extent, crs=ccrs.PlateCarree())
# ax.coastlines()
# plt.show()
