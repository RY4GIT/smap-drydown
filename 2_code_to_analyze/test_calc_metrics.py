# Import libraries
import json
import requests
import os
from ismn.interface import ISMN_Interface
from ismn.meta import Depth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # matplotlib is not installed automatically
from datetime import datetime
import warnings

# Specify current directory and create output directory if it does not exist
os.chdir("G:/Shared drives/Ryoko and Hilary/SMSigxSMAP/analysis/0_code")

class SMAPxISMN():
    def __init__(self, input_path_ismn, input_path_smap, insitu_network_name, out_path):
        self.input_path_ismn = input_path_ismn
        self.input_path_smap = input_path_smap
        self.insitu_network_name = insitu_network_name
        self.out_path = out_path

    def load_data(self, plot=True):

        #################
        #   READ DATA   #
        #################

        # Read SMAP data
        df_SMAP = pd.read_csv(self.input_path_smap)
        df_ts_smap = df_SMAP[df_SMAP['ID']==0].copy() #TODO: make this more generic
        df_ts_smap['Date'] = pd.to_datetime(df_ts_smap['Date'])
        df_ts_smap.set_index('Date', inplace=True)

        # Read ISMN data
        ismn_data = ISMN_Interface(self.input_path_ismn)

        # Find the ISMN station corresponding to the SMAP extraction
        ismn_grid = ismn_data.collection.grid
        gpis, lons, lats = ismn_grid.get_grid_points()
        station, dist = ismn_data.collection.get_nearest_station(df_SMAP['Longitude'][0], df_SMAP['Latitude'][0])
        print(f'Station {station.name} is {int(dist)} metres away from the passed coordinates:')
        if int(dist) >= 100:
            warnings.warn('The sensor is too far away')

        # Get the ISMN sensor installed shallower than 0.05 cm
        for sensor in station.iter_sensors(variable='soil_moisture',depth=Depth(0., 0.05)):
            print(sensor)
            df_ts_ismn = sensor.read_data()

        ###################################
        #   QUALITY CONTROL OF THE DATA   #
        ###################################

        # ISMN
        bad_data_idx_ismn = df_ts_ismn[df_ts_ismn['soil_moisture_flag']!='G'].index
        df_ts_ismn.drop(bad_data_idx_ismn, inplace=True)
        df_ts_ismn_daily = df_ts_ismn.resample('D', axis=0).mean()

        # SMAP
        """
        # 1. Use the retrieval_qual_flag field to identify retrievals in the soil_moisture field estimated to be of 
        recommended quality. A retrieval_qual_flag value of either 0 or 8 indicates high-quality retrievals 
        (8 because a failed F/T retrieval does not affect soil moisture retrieval). 
        """

        df_ts_smap_am = df_ts_smap[['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_soil_moisture', 'SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag']].copy()
        bad_data_idx_smap = df_ts_smap_am[(df_ts_smap_am['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'] != 0.0) & (df_ts_smap_am['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag'] != 8.0)].index
        df_ts_smap_am.drop(bad_data_idx_smap, inplace=True)
        del bad_data_idx_smap

        df_ts_smap_pm = df_ts_smap[['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm', 'SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm']].copy()
        bad_data_idx_smap = df_ts_smap_pm[(df_ts_smap_pm['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm'] != 0.0) & (df_ts_smap_pm['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm'] != 8.0)].index
        df_ts_smap_pm.drop(bad_data_idx_smap, inplace=True)
        del bad_data_idx_smap

        #########################################
        #   MERGE DATA AND AGGREGATE TO DAILY   #
        #########################################

        # SMAP, get the average of AM and PM data
        df_ts_smap_daily = pd.merge(df_ts_smap_am, df_ts_smap_pm, how='inner', left_index=True, right_index=True)
        df_ts_smap_daily['soil_moisture_smap'] = df_ts_smap_daily[['SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_AM_soil_moisture','SPL3SMP_E_005_Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm']].mean(axis=1, skipna=True)
        df_ts_smap_daily = df_ts_smap_daily['soil_moisture_smap'].resample('D', axis=0).mean()

        # Merge and sync ISMN and SMAP data
        df_ts_sync = pd.merge(df_ts_smap_daily, df_ts_ismn_daily, how='inner', left_index=True, right_index=True)
        df_ts_sync.rename(columns={"soil_moisture": "soil_moisture_ismn"}, inplace=True)

        # Filter out the portion of timeseries where either ISMN or SMAP data is unavailable
        df_ts_sync['rolling_ismn'] = df_ts_sync['soil_moisture_ismn'].rolling(window='7d', center=True, closed='both').mean()
        df_ts_sync['rolling_smap'] = df_ts_sync['soil_moisture_smap'].rolling(window='7d', center=True, closed='both').mean()
        no_data_idx = df_ts_sync[df_ts_sync['rolling_ismn'].isnull() | df_ts_sync['rolling_smap'].isnull()].index
        df_ts_sync['soil_moisture_ismn'][no_data_idx]=np.NaN
        df_ts_sync['soil_moisture_smap'][no_data_idx]=np.NaN

        self.sm_ts = df_ts_sync

        if plot:
            # Plot the timesereis of data

            fig, ax = plt.subplots()
            line1, = ax.plot(df_ts_sync['soil_moisture_ismn'], label='In-situ')
            line2, = ax.plot(df_ts_sync['soil_moisture_smap'], 'o', markersize=4, alpha=0.5, label='SMAP')
            fig.legend()
            xax = ax.xaxis
            ax.set_title(f"{station.name} station, OZNET, Australia\n({df_SMAP['Longitude'][0]}, {df_SMAP['Latitude'][0]})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Volumetric soil water content [m^3/m^3]")
            # formatter = mdates.DateFormatter("%Y") ### formatter of the date
            # locator = mdates.YearLocator() ### where to put the labels
            # ax.xaxis.set_major_formatter(formatter) ## calling the formatter for the x-axis
            # ax.xaxis.set_major_locator(locator) ## calling the locator for the x-axis
            fig.autofmt_xdate()
            fig.savefig(os.path.join(self.out_path, 'test_ts.png'))
            del fig, ax

            # Plot the histogram
            fig, ax = plt.subplots()
            ax.hist(df_ts_sync['soil_moisture_ismn'], label='In-situ',  bins=30, alpha = 0.5, density=True, stacked=True)
            ax.hist(df_ts_sync['soil_moisture_smap'], label='SMAP',     bins=30, alpha = 0.5, density=True, stacked=True)
            fig.legend()
            ax.set_title(f"{station.name} station, OZNET, Australia\n({df_SMAP['Longitude'][0]}, {df_SMAP['Latitude'][0]})")
            ax.set_xlabel("Volmetric soil water content [m^3/m^3]")
            ax.set_ylabel("Normalized frequency [-]")
            fig.savefig(os.path.join(self.out_path, 'test_hist.png'))
            del fig, ax


def main():

    input_path_smap = "../1_data/SMAP/OZNET/Point-Example-SPL3SMP-E-005-results.csv"
    input_path_ismn = "G:/Shared drives/Ryoko and Hilary/SMSigxISMN/analysis/0_data_raw/ISMN"
    out_path = "../3_data_out/"

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    mySMAP = SMAPxISMN(input_path_smap=input_path_smap, input_path_ismn=input_path_ismn, insitu_network_name='OZNET', out_path=out_path)
    mySMAP.load_data()

if __name__ == '__main__':
    main()




# TODO: check the timezone UTC? local time?
# All measurements are converted to Coordinated Universal Time (UTC).
# TODO: generalize more