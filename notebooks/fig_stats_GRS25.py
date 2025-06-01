# %% Import packages
import os
import sys
import getpass
import json

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, spearmanr, mannwhitneyu, ks_2samp, median_test
from scipy.interpolate import griddata
import statsmodels.api as statsm
from functions import q_model, loss_model

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import (
    Normalize,
    LinearSegmentedColormap,
    ListedColormap,
    BoundaryNorm,
)
import cartopy.crs as ccrs
from textwrap import wrap

# !pip install mpl-scatter-density
import mpl_scatter_density

# Math font
import matplotlib as mpl

plt.rcParams["font.family"] = "DejaVu Sans"  # Or any other available font
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # Ensure the font is set correctly
# mpl.rcParams["font.family"] = "sans-serif"
# mpl.rcParams["font.sans-serif"] = "Myriad Pro"
mpl.rcParams["font.size"] = 12.0
mpl.rcParams["axes.titlesize"] = 12.0
plt.rcParams["mathtext.fontset"] = (
    "stixsans"  #'stix'  # Or 'cm' (Computer Modern), 'stixsans', etc.
)

# Ryoko do not have this font on my system
# mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# font_files = mpl.font_manager.findSystemFonts(fontpaths=['/home/brynmorgan/Fonts/'])
# for font_file in font_files:
#     mpl.font_manager.fontManager.addfont(font_file)

# %% Plot config

################ CHANGE HERE FOR PATH CONFIG ##################################
############ CHANGE HERE FOR CHECKING DIFFERENT RESULTS ###################
dir_name = f"raraki_2024-12-03_revision"  # f"raraki_2024-05-13_global_piecewise"  # "raraki_2024-02-02"  # f"raraki_2023-11-25_global_95asmax"
###########################################################################
# f"raraki_2024-05-13_global_piecewise" was used for the 1st version of the manuscript

# Ryoko Araki, Bryn Morgan, Hilary K McMillan, Kelly K Caylor.
# Nonlinear Soil Moisture Loss Function Reveals Vegetation Responses to Water Availability. ESS Open Archive . August 01, 2024.
# DOI: 10.22541/essoar.172251989.99347091/v1

# Data dir
user_name = getpass.getuser()
data_dir = rf"/home/{user_name}/waves/projects/smap-drydown/data"

# Read the model output (results)
output_dir = rf"/home/{user_name}/waves/projects/smap-drydown/output"
results_file = rf"all_results_processed_copy.csv"

datarods_dir = "datarods"
anc_rangeland_processed_file = "anc_info_rangeland_processed.csv"
coord_info_file = "coord_info.csv"

################ CHANGE HERE FOR PLOT VISUAL CONFIG & SETTING VARIABLES #########################

## Define parameters
save = True

note_dir = rf"/home/{user_name}/smap-drydown/notebooks"
with open(os.path.join(note_dir, "fig_veg_colors_lim.json"), "r") as file:
    vegetation_color_dict = json.load(file)

# Load variable settings
with open(os.path.join(note_dir, "fig_variable_labels.json"), "r") as file:
    var_dict = json.load(file)

# %% ############################################################################
# Create figure output directory in the model output directory
fig_dir = os.path.join(output_dir, dir_name, "figs_GRC25")
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
    print(f"Created dir: {fig_dir}")
else:
    print(f"Already exists: {fig_dir}")

# Open the output file
f = open(os.path.join(fig_dir, "log.txt"), "w")
original_stdout = sys.stdout  # Save the original stdout
sys.stdout = f  # Change the stdout to the file handle

# ############################################################################
# DATA IMPORT

df = pd.read_csv(os.path.join(output_dir, dir_name, results_file))
print("Loaded results file")

coord_info = pd.read_csv(os.path.join(data_dir, datarods_dir, coord_info_file))

# Get bins for ancillary data
sand_bin_list = [i * 0.1 for i in range(11)]
sand_bin_list = sand_bin_list[1:]
sand_cmap = "Oranges_r"

df["sand_bins"] = pd.cut(df["sand_fraction"], bins=sand_bin_list, include_lowest=True)
first_I = df["sand_bins"].cat.categories[0]
new_I = pd.Interval(0.1, first_I.right)
df["sand_bins"] = df["sand_bins"].cat.rename_categories({first_I: new_I})

# ai
ai_bin_list = [i * 0.25 for i in range(7)]
ai_cmap = "RdBu"

df["ai_bins"] = pd.cut(df["AI"], bins=ai_bin_list, include_lowest=True)
first_I = df["ai_bins"].cat.categories[0]
new_I = pd.Interval(0, first_I.right)
df["ai_bins"] = df["ai_bins"].cat.rename_categories({first_I: new_I})

# %%
print(fig_dir)
# %% ###################################################
# Exclude model fits failure
def count_median_number_of_events_perGrid(df):
    grouped = df.groupby(["EASE_row_index", "EASE_column_index"]).agg(
        median_diff_R2_q_tauexp=("diff_R2_q_tauexp", "median"),
        count=("diff_R2_q_tauexp", "count"),
    )
    print(f"Median number of drydowns per SMAP grid: {grouped['count'].median()}\n")


print(f"Total number of events: {len(df)}")
count_median_number_of_events_perGrid(df)

###################################################
# Defining model acceptabiltiy criteria
R2_thresh = 0.8
sm_range_thresh = 0.20
small_q_thresh = 1.0e-04
large_q_thresh = 0.8
###################################################


def filter_df(df, criteria):
    return df[criteria].copy()


def print_model_success(message, df):
    print(f"{message} {len(df)}")
    count_median_number_of_events_perGrid(df)


# Filtering dfframes based on various model criteria
criteria_q = (df["q_r_squared"] > R2_thresh) & (df["sm_range"] > sm_range_thresh)
criteria_specific_q = (
    criteria_q
    & (df["q_q"] > small_q_thresh)
    & (df["large_q_criteria"] < large_q_thresh)
    & (df["first3_avail2"])
)
criteria_tauexp = (df["tauexp_r_squared"] > R2_thresh) & (
    df["sm_range"] > sm_range_thresh
)
criteria_exp = (df["exp_r_squared"] > R2_thresh) & (df["sm_range"] > sm_range_thresh)

df_filt_q = filter_df(df, criteria_specific_q)
df_filt_allq = filter_df(df, criteria_q)
df_filt_tauexp = filter_df(df, criteria_tauexp)
df_filt_exp = filter_df(df, criteria_exp)
df_filt_q_or_tauexp = filter_df(df, criteria_q | criteria_tauexp)
df_filt_q_and_tauexp = filter_df(df, criteria_q & criteria_tauexp)
df_filt_q_or_exp = filter_df(df, criteria_q | criteria_exp)
df_filt_q_and_exp = filter_df(df, criteria_q & criteria_exp)

# Printing success messages and calculating events
print_model_success("q model fit successful:", df_filt_q)
print_model_success("exp model fit successful:", df_filt_exp)
print_model_success("tau-exp model fit successful:", df_filt_tauexp)
print_model_success("both q and exp model fit successful:", df_filt_q_and_exp)
print_model_success("both q and tau-exp model fit successful:", df_filt_q_and_tauexp)
# print_model_success("either q or tau-exp:", df_filt_q_or_tauexp)
# print_model_success("either q or exp model fit successful:", df_filt_q_or_exp)
print_model_success(
    "q model fit successful (not filtering q parameter values):", df_filt_allq
)


def print_performance_comparison(df, model1, model2):
    n_better = sum(df[model1] > df[model2])
    percentage_better = n_better / len(df) * 100
    print(
        f"Of successful fits, {model1} performed better in {percentage_better:.1f} percent of events: {n_better}"
    )


print_performance_comparison(df_filt_q_and_tauexp, "q_r_squared", "tauexp_r_squared")
print_performance_comparison(df_filt_q_and_exp, "q_r_squared", "exp_r_squared")


# %%
########################################################################
# # Group by pixel (EASE_row_index, EASE_column_index)


def most_frequent_str(series):
    return series.mode()[0] if not series.mode().empty else None


df_filt_q_agg = (
    df_filt_q.groupby(["EASE_row_index", "EASE_column_index"])
    .agg(
        {
            "q_q": ["median", "var"],  # Calculate median and variance of "q_q"
            "AI": "median",  # Calculate median of "AI",
            "sand_fraction": "median",
            "name": most_frequent_str,
            "q_ETmax": ["median", "var"],
            "q_theta_star": ["median", "var"],
            "q_theta_w": ["median", "var"],
            "event_length": "median",
            "id_x": "count",  # Calculate count of rows for each group
        }
    )
    .reset_index()
)

# Flatten the multi-level column index
df_filt_q_agg.columns = [
    "_".join(col).strip() if col[1] else col[0] for col in df_filt_q_agg.columns.values
]
df_filt_q_agg.head()

df_filt_q_agg["sand_bins"] = pd.cut(
    df_filt_q_agg["sand_fraction_median"], bins=sand_bin_list, include_lowest=True
)
first_I = df["sand_bins"].cat.categories[0]
new_I = pd.Interval(0.1, first_I.right)
df_filt_q_agg["sand_bins"] = df_filt_q_agg["sand_bins"].cat.rename_categories(
    {first_I: new_I}
)

df_filt_q_agg["ai_bins"] = pd.cut(
    df_filt_q_agg["AI_median"], bins=ai_bin_list, include_lowest=True
)
first_I = df_filt_q_agg["ai_bins"].cat.categories[0]
new_I = pd.Interval(0, first_I.right)
df_filt_q_agg["ai_bins"] = df_filt_q_agg["ai_bins"].cat.rename_categories(
    {first_I: new_I}
)

# df_filt_q_agg = df_filt_q_agg[df_filt_q_agg["id_x_count"]>8]

# %%
####################################################################################
# Read rangeland df and join it with d_filt_q
_rangeland_info = pd.read_csv(
    os.path.join(data_dir, datarods_dir, anc_rangeland_processed_file)
).drop(["Unnamed: 0"], axis=1)
# Change with fraction to percentage
_rangeland_info["fractional_wood"] = _rangeland_info["fractional_wood"] * 100
_rangeland_info["fractional_herb"] = _rangeland_info["fractional_herb"] * 100
rangeland_info = _rangeland_info.merge(
    coord_info, on=["EASE_row_index", "EASE_column_index"]
)

# merge with results dfframe
df_filt_q_conus = df_filt_q.merge(
    rangeland_info, on=["EASE_row_index", "EASE_column_index", "year"], how="left"
)


# %%
def save_figure(fig, fig_dir, filename, save_format, dpi):
    path = os.path.join(fig_dir, f"{filename}.{save_format}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close()


# %%
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
#
# Plots & Stats
#
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

###################################################################
# Number of samples
################################################################0
# How much percent area (based on SMAP pixels) had better R2


def count_model_performance(df, varname):
    grouped = df.groupby(["EASE_row_index", "EASE_column_index"]).agg(
        median_diff_R2=(varname, "median"), count=(varname, "count")
    )
    print(f"Median number of drydowns per SMAP grid: {grouped['count'].median()}")
    print(f"Number of SMAP grids with df: {len(grouped)}")
    pos_median_diff_R2 = (grouped["median_diff_R2"] > 0).sum()
    print(
        f"Number of SMAP grids with bettter nonlinear model fits: {pos_median_diff_R2} ({(pos_median_diff_R2/len(grouped))*100:.1f} percent)"
    )
    # sns.histplot(grouped["count"], binwidth=0.5, color="#2c7fb8", fill=False, linewidth=3)


count_model_performance(df_filt_q_or_exp, "diff_R2_q_tauexp")
count_model_performance(df_filt_q_or_exp, "diff_R2_q_exp")

# %%
###################################################################
# Number of samples
###################################################################
sample_sand_stat = df_filt_q[["id_x", "sand_bins"]].groupby("sand_bins").count()
print(sample_sand_stat)
sample_sand_stat.to_csv(os.path.join(fig_dir, f"sample_sand_stat.csv"))

sample_ai_stat = df_filt_q[["id_x", "ai_bins"]].groupby("ai_bins").count()
print(sample_ai_stat)
sample_ai_stat.to_csv(os.path.join(fig_dir, f"sample_ai_stat.csv"))

sample_veg_stat = df_filt_q[["id_x", "name"]].groupby("name").count()
print(sample_veg_stat)
sample_veg_stat.to_csv(os.path.join(fig_dir, f"sample_veg_stat.csv"))

# Check no df in sand
print(sum(pd.isna(df_filt_q["sand_fraction"]) == True))

# %%
############################################################################
# PLOTTING FUNCTION STARTS HERE
###########################################################################


############################################################################
# Map plots
###########################################################################
def plot_map(
    ax, df, coord_info, cmap, norm, var_item, stat_type, title="", bar_label=None
):
    plt.setp(ax.spines.values(), linewidth=0.5)

    # Get the mean values of the variable
    if stat_type == "median":
        stat = df.groupby(["EASE_row_index", "EASE_column_index"])[
            var_item["column_name"]
        ].median()
        stat_label = "Median"
    elif stat_type == "mean":
        stat = df.groupby(["EASE_row_index", "EASE_column_index"])[
            var_item["column_name"]
        ].mean()
        stat_label = "Mean"

    # Reindex to the full EASE row/index extent
    new_index = pd.MultiIndex.from_tuples(
        zip(coord_info["EASE_row_index"], coord_info["EASE_column_index"]),
        names=["EASE_row_index", "EASE_column_index"],
    )
    stat_pad = stat.reindex(new_index, fill_value=np.nan)

    # Join latitude and longitude
    merged_df = (
        stat_pad.reset_index()
        .merge(
            coord_info[
                ["EASE_row_index", "EASE_column_index", "latitude", "longitude"]
            ],
            on=["EASE_row_index", "EASE_column_index"],
            how="left",
        )
        .set_index(["EASE_row_index", "EASE_column_index"])
    )

    # Create pivot array
    pivot_array = merged_df.pivot(
        index="latitude", columns="longitude", values=var_item["column_name"]
    )
    pivot_array[pivot_array.index > -60]  # Exclude antarctica in the map (no df)

    # Get lat and lon
    lons = pivot_array.columns.values
    lats = pivot_array.index.values

    # Plot in the map
    im = ax.pcolormesh(
        lons, lats, pivot_array, norm=norm, cmap=cmap, transform=ccrs.PlateCarree()
    )
    ax.set_extent([-160, 170, -60, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)

    if not bar_label:
        bar_label = f'{var_item["symbol"]} {var_item["unit"]}'

    # Add colorbar
    plt.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        # label=f'{stat_label} {var_item["label"]} {var_item["unit"]}',
        label=bar_label,
        shrink=0.25,
        # width=0.1,
        pad=0.02,
    )

    # Set plot title and labels
    # ax.set_title(f'Mean {variable_name} per pixel')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if title != "":
        ax.set_title(title, loc="left")


stat_type = "median"


def print_global_stats(df, diff_var, model_desc):
    print(f"Global median diff R2 ({model_desc}): {df[diff_var].median()}")
    print(f"Global mean diff R2 ({model_desc}): {df[diff_var].mean()}")


# Setup common variables
var_key_exp = "diff_R2_exp"
var_key_tauexp = "diff_R2_tauexp"
norm_exp = Normalize(
    vmin=var_dict[var_key_exp]["lim"][0], vmax=var_dict[var_key_exp]["lim"][1]
)
norm_tauexp = Normalize(
    vmin=var_dict[var_key_tauexp]["lim"][0], vmax=var_dict[var_key_tauexp]["lim"][1]
)

# Plot and save maps for exp model
plt.rcParams.update({"font.size": 12})
fig_map_R2, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_exp,
    coord_info=coord_info,
    cmap="RdBu",
    norm=norm_exp,
    var_item=var_dict[var_key_exp],
    stat_type=stat_type,
    bar_label=var_dict[var_key_exp]["label"],
)
if save:
    save_figure(fig_map_R2, fig_dir, f"R2_map_{stat_type}_and_exp", "png", 900)

# Print statistical summaries for exp model
print_global_stats(df_filt_q_and_exp, "diff_R2_q_exp", "nonlinear - linear")

# Plot and save maps for tauexp model
fig_map_R2, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_tauexp,
    coord_info=coord_info,
    cmap="RdBu",
    norm=norm_tauexp,
    var_item=var_dict[var_key_tauexp],
    stat_type=stat_type,
    bar_label=var_dict[var_key_tauexp]["label"],
)
if save:
    save_figure(fig_map_R2, fig_dir, f"R2_map_{stat_type}_and_tauexp", "png", 900)

# Print statistical summaries for tauexp model
print_global_stats(
    df_filt_q_and_tauexp, "diff_R2_q_tauexp", "nonlinear - tau-based linear"
)

# %% Map of q
plt.rcParams.update({"font.size": 12})
var_key = "q_q"

q_colors = [
    "#F7CA0D",
    "#91cf60",
    "#01665e",
]  # These are your colors c1, c2, and c3 # "#5ab4ac",
q_cmap = LinearSegmentedColormap.from_list("custom_cmap", q_colors, N=256)
norm = Normalize(vmin=0.75, vmax=3.0)
fig_map_q, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()})
stat_type = "median"
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap=q_cmap,
    norm=norm,
    var_item=var_dict[var_key],
    stat_type=stat_type,
)

save_figure(fig_map_q, fig_dir, f"q_map_{stat_type}", "png", 900)
# save_figure(fig_map_q, fig_dir, f"q_map_{stat_type}", "pdf", 1200)

print(f"Global median q: {df_filt_q['q_q'].median()}")
print(f"Global mean q: {df_filt_q['q_q'].mean()}")
print(f"Global q<1 median: {df_filt_q[df_filt_q["q_q"] < 1]["q_q"].median():.2f}")
print(f"Global q<1 mean: {df_filt_q[df_filt_q["q_q"] < 1]["q_q"].mean():.2f}")
print(f"Global q>1 median: {df_filt_q[df_filt_q["q_q"] > 1]["q_q"].median():.2f}")
print(f"Global q<1 mean: {df_filt_q[df_filt_q["q_q"] > 1]["q_q"].mean():.2f}")
# %%
# Map of theta_star
var_key = "q_theta_star"
norm = Normalize(vmin=0.1, vmax=0.4)
fig_map_theta_star, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="median",
    title="(a)",
)
plt.show()
if save:
    save_figure(fig_map_theta_star, fig_dir, f"sup_map_{var_key}", "png", 900)

print(f"Global median theta_star: {df_filt_q['max_sm'].median()}")
print(f"Global mean theta_star: {df_filt_q['max_sm'].mean()}")

# %%
# Map of ETmax
var_key = "q_ETmax"
norm = Normalize(vmin=0, vmax=6)
fig_map_ETmax, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q,
    coord_info=coord_info,
    cmap="YlGnBu",
    norm=norm,
    var_item=var_dict[var_key],
    stat_type="median",
    title="(b)",
)
plt.show()
if save:
    save_figure(fig_map_ETmax, fig_dir, f"sup_map_{var_key}", "png", 900)

print(f"Global median ETmax: {df_filt_q['q_ETmax'].median()}")
print(f"Global mean ETmax: {df_filt_q['q_ETmax'].mean()}")


# %%
############################################################################
# Histogram of q values (global)
###########################################################################


def plot_hist(df, var_key):
    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Create the histogram with a bin width of 1
    sns.histplot(
        df[var_key], binwidth=0.2, color="#62AD5F", fill=False, linewidth=3, ax=ax
    )

    # Calculate median and mean
    median_value = df[var_key].median()
    mean_value = df[var_key].mean()

    # Add median and mean as vertical lines
    ax.axvline(
        median_value, color="tab:grey", linestyle="--", linewidth=3, label=f"Median"
    )
    ax.axvline(mean_value, color="tab:grey", linestyle=":", linewidth=3, label=f"Mean")

    # Setting the x limit
    ax.set_xlim(0, 6)

    # Adding title and labels
    ax.set_xlabel(var_dict[var_key]["symbol"])
    ax.set_ylabel("Frequency")
    fig.legend(loc="upper right", bbox_to_anchor=(0.93, 0.9), fontsize="small")

    return fig, ax


plt.rcParams.update({"font.size": 30})
fig_q_hist, _ = plot_hist(df=df_filt_q, var_key="q_q")
if save:
    save_figure(fig_q_hist, fig_dir, f"q_hist", "png", 900)
    save_figure(fig_q_hist, fig_dir, f"q_hist", "pdf", 1200)


# %%
############################################################################
# Loss function plot + Scatter plots with error bars
###########################################################################
def wrap_text(text, width):
    return "\n".join(wrap(text, width))


def plot_loss_func(
    ax,
    df,
    z_var,
    categories=None,
    colors=None,
    cmap=None,
    title="",
    plot_legend=False,
    median_by_pixel=False,
):

    # Get category/histogram bins to enumerate
    if categories is None:
        # Get unique bins
        bins_in_range = df[z_var["column_name"]].unique()
        bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
        bins_sorted = sorted(bins_list, key=lambda x: x.left)

        # Get colors according to the number of bins
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / len(bins_sorted)) for i in range(len(bins_sorted))]
    else:
        bins_sorted = categories

    # For each row in the subset, calculate the loss for a range of theta values
    for i, category in enumerate(bins_sorted):
        # Get subset
        subset = df[df[z_var["column_name"]] == category]

        # Get the median of all the related loss function parameters
        if not median_by_pixel:
            theta_w = subset["q_theta_w"].median()
            theta_star = subset["q_theta_star"].median()
            ETmax = subset["q_ETmax"].median()
            q = subset["q_q"].median()
        else:
            theta_w = subset["q_theta_w_median"].median()
            theta_star = subset["q_theta_star_median"].median()
            ETmax = subset["q_ETmax_median"].median()
            q = subset["q_q_median"].median()

        # Calculate the loss function
        theta = np.arange(theta_w, theta_star, 0.01)
        dtheta = loss_model(
            theta=theta, q=q, ETmax=ETmax, theta_w=theta_w, theta_star=theta_star
        )

        # Plot loss function from median parameters
        ax.plot(
            theta,
            dtheta,
            label=f"{category}",
            color=colors[i],
            linewidth=3,
        )

    ax.invert_yaxis()
    ax.set_xlabel(
        f"{var_dict['theta']['label']}\n{var_dict['theta']['symbol']} {var_dict['theta']['unit']}"
    )
    ax.set_ylabel(r"$d\theta/dt$" + f" {var_dict['dtheta']['unit']}")
    if title == "":
        title = f'Median loss function by {z_var["label"]} {z_var["unit"]}'
    ax.set_title(title, loc="left", fontsize=14)

    if plot_legend:
        if categories is None:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                title=f'{z_var["label"]}\n{z_var["unit"]}',
            )
        else:
            legend = ax.legend(bbox_to_anchor=(1, 1))
            for text in legend.get_texts():
                label = text.get_text()
                wrapped_label = wrap_text(label, 16)  # Wrap text after 16 characters
                text.set_text(wrapped_label)


def plot_scatter_with_errorbar(
    ax,
    df,
    x_var,
    y_var,
    z_var,
    quantile,
    cmap=None,
    categories=None,
    colors=None,
    title="",
    plot_logscale=False,
    plot_legend=False,
):

    stats_dict = {}

    if categories is None:
        # Get unique bins
        bins_in_range = df[z_var["column_name"]].unique()
        bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
        bins_sorted = sorted(bins_list, key=lambda x: x.left)
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / len(bins_sorted)) for i in range(len(bins_sorted))]
    else:
        bins_sorted = categories

    # Calculate median and 90% confidence intervals for each vegetation class
    for i, category in enumerate(bins_sorted):
        subset = df[df[z_var["column_name"]] == category]

        # Median calculation
        x_median = subset[x_var["column_name"]].median()
        y_median = subset[y_var["column_name"]].median()

        # Confidence interval calculation
        x_ci_low, x_ci_high = np.nanpercentile(
            subset[x_var["column_name"]], [quantile, 100 - quantile]
        )
        y_ci_low, y_ci_high = np.nanpercentile(
            subset[y_var["column_name"]], [quantile, 100 - quantile]
        )

        # Store in dict
        stats_dict[category] = {
            "x_median": x_median,
            "y_median": y_median,
            "x_ci": (x_median - x_ci_low, x_ci_high - x_median),
            "y_ci": (y_median - y_ci_low, y_ci_high - y_median),
            "color": colors[i],
        }

    # Now plot medians with CIs
    for category, stats in stats_dict.items():
        ax.errorbar(
            stats["x_median"],
            stats["y_median"],
            xerr=np.array([[stats["x_ci"][0]], [stats["x_ci"][1]]]),
            yerr=np.array([[stats["y_ci"][0]], [stats["y_ci"][1]]]),
            fmt="o",
            label=str(category),
            capsize=5,
            capthick=2,
            color=stats["color"],
            alpha=0.7,
            markersize=10,
            mec=stats["color"],
            mew=1,
            linewidth=3,
        )

    # Add labels and title
    ax.set_xlabel(f"{x_var['symbol']} {x_var['unit']}")
    ax.set_ylabel(f"{y_var['symbol']} {y_var['unit']}")
    if title == "":
        title = f"Median with {quantile}% confidence interval"

    ax.set_title(title, loc="left", fontsize=14)

    # Add a legend
    if plot_legend:
        plt.legend(bbox_to_anchor=(1, 1.5))
    if plot_logscale:
        plt.xscale("log")
    ax.set_xlim(x_var["lim"][0], x_var["lim"][1])
    ax.set_ylim(y_var["lim"][0], y_var["lim"][1])


# %%
#####################################
#  Fig 4
# %%
# Aridity
plt.rcParams.update({"font.size": 16})
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plot_loss_func(
    axs[0],
    df=df_filt_q_agg,
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    plot_legend=False,
    title="(a)",
    median_by_pixel=True,
)

plot_scatter_with_errorbar(
    ax=axs[1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title=" ",
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"fig4_lossfnc_ai", "png", 1200)
    save_figure(fig, fig_dir, f"fig4_lossfnc_ai", "pdf", 1200)

# %%
# Sand
plt.rcParams.update({"font.size": 16})
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plot_loss_func(
    ax=axs[0],
    df=df_filt_q_agg,
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    plot_legend=False,
    title="(b)",
    median_by_pixel=True,
)

plot_scatter_with_errorbar(
    ax=axs[1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title=" ",
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"fig4_lossfnc_sand", "png", 1200)
    save_figure(fig, fig_dir, f"fig4_lossfnc_sand", "pdf", 1200)

# %%
# Vegetation
plt.rcParams.update({"font.size": 16})
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plot_loss_func(
    axs[0],
    df_filt_q_agg,
    var_dict["veg_class_mode"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    plot_legend=False,
    title="(c)",
    median_by_pixel=True,
)

plot_scatter_with_errorbar(
    ax=axs[1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["veg_class_mode"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title=" ",
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"fig4_lossfnc_veg", "png", 1200)
    save_figure(fig, fig_dir, f"fig4_lossfnc_veg", "pdf", 1200)
# %%
#####################################
#  4-grid Loss function plots + parameter scatter plots
#######################################
# Vegetation
plt.rcParams.update({"font.size": 16})
fig, axs = plt.subplots(2, 2, figsize=(8, 7.5))
plot_loss_func(
    axs[0, 0],
    df_filt_q,
    var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    plot_legend=False,
    title="(a)",
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q,
    x_var=var_dict["q_theta_star"],
    y_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="(b)",
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="(c)",
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q,
    x_var=var_dict["q_theta_star"],
    y_var=var_dict["q_ETmax"],
    z_var=var_dict["veg_class"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="(d)",
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"sup_lossfnc_veg", "png", 1200)
    save_figure(fig, fig_dir, f"sup_lossfnc_veg", "pdf", 1200)

# # %%
# # With median
plt.rcParams.update({"font.size": 18})
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plot_loss_func(
    axs[0, 0],
    df_filt_q_agg,
    var_dict["veg_class_mode"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    plot_legend=False,
    title="(a)",
    median_by_pixel=True,
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["veg_class_mode"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="(b)",
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q_agg,
    x_var=var_dict["q_ETmax_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["veg_class_mode"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="(c)",
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_ETmax_median"],
    z_var=var_dict["veg_class_mode"],
    quantile=25,
    categories=list(vegetation_color_dict.keys()),
    colors=list(vegetation_color_dict.values()),
    title="(d)",
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"sup_lossfnc_veg_median_by_pixel", "png", 1200)
    save_figure(fig, fig_dir, f"sup_lossfnc_veg_median_by_pixel", "pdf", 1200)

# %%
# Aridity Index
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.rcParams.update({"font.size": 16})

plot_loss_func(
    ax=axs[0, 0],
    df=df_filt_q,
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    plot_legend=False,
    title="(a)",
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q,
    x_var=var_dict["q_theta_star"],
    y_var=var_dict["q_q"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="(b)",
    plot_logscale=False,
    plot_legend=False,
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["q_q"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="(c)",
    plot_logscale=False,
    plot_legend=False,
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q,
    x_var=var_dict["q_theta_star"],
    y_var=var_dict["q_ETmax"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="(d)",
    plot_logscale=False,
    plot_legend=False,
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"sup_lossfnc_ai", "png", 1200)
    save_figure(fig, fig_dir, f"sup_lossfnc_ai", "pdf", 1200)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.rcParams.update({"font.size": 16})

plot_loss_func(
    ax=axs[0, 0],
    df=df_filt_q_agg,
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    plot_legend=False,
    title="(a)",
    median_by_pixel=True,
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="(b)",
    plot_logscale=False,
    plot_legend=False,
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q_agg,
    x_var=var_dict["q_ETmax_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="(c)",
    plot_logscale=False,
    plot_legend=False,
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_ETmax_median"],
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    quantile=25,
    title="(d)",
    plot_logscale=False,
    plot_legend=False,
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"sup_lossfnc_ai_median_by_pixel", "png", 1200)
    save_figure(fig, fig_dir, f"sup_lossfnc_ai_median_by_pixel", "pdf", 1200)

# %%
# sand
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

plot_loss_func(
    ax=axs[0, 0],
    df=df_filt_q,
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    plot_legend=False,
    title="(a)",
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q,
    x_var=var_dict["q_theta_star"],
    y_var=var_dict["q_q"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="(b)",
    plot_logscale=False,
    plot_legend=False,
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q,
    x_var=var_dict["q_ETmax"],
    y_var=var_dict["q_q"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="(c)",
    plot_logscale=False,
    plot_legend=False,
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q,
    x_var=var_dict["q_theta_star"],
    y_var=var_dict["q_ETmax"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="(d)",
    plot_logscale=False,
    plot_legend=False,
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"sup_lossfnc_sand", "png", 1200)
    save_figure(fig, fig_dir, f"sup_lossfnc_sand", "pdf", 1200)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

plot_loss_func(
    ax=axs[0, 0],
    df=df_filt_q_agg,
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    plot_legend=False,
    title="(a)",
    median_by_pixel=True,
)

plot_scatter_with_errorbar(
    ax=axs[0, 1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="(b)",
    plot_logscale=False,
    plot_legend=False,
)

plot_scatter_with_errorbar(
    ax=axs[1, 0],
    df=df_filt_q_agg,
    x_var=var_dict["q_ETmax_median"],
    y_var=var_dict["q_q_median"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="(c)",
    plot_logscale=False,
    plot_legend=False,
)
plot_scatter_with_errorbar(
    ax=axs[1, 1],
    df=df_filt_q_agg,
    x_var=var_dict["q_theta_star_median"],
    y_var=var_dict["q_ETmax_median"],
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    quantile=25,
    title="(d)",
    plot_logscale=False,
    plot_legend=False,
)

plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"sup_lossfnc_sand_median_by_pixel", "png", 1200)
    save_figure(fig, fig_dir, f"sup_lossfnc_sand_median_by_pixel", "pdf", 1200)


# %%
# ###########################################################
# ###########################################################
#                    Rangeland plots
# ###########################################################
# ###########################################################
# Get the statistics on the proportion of q>1 and q<1 events
def get_df_percentage_q(
    df,
    x1_varname,
    x2_varname,
    y_varname,
    weight_by,
    bins=[0, 20, 40, 60, 80, 100],
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
):

    # Bin AI values
    x2_new_varname = x2_varname + "_binned2"
    df[x2_new_varname] = pd.cut(
        df[x2_varname],
        # bins=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, np.inf],
        # labels=["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1.0","1.0-1.25", "1.25-1.5", "1.5-"],
        bins=[0, 0.5, 1.0, 1.5, np.inf],
        labels=["0-0.5", "0.5-1.0", "1.0-1.5", "1.5-"],
    )

    x1_new_varname = x1_varname + "_pct"
    df[x1_new_varname] = pd.cut(df[x1_varname], bins=bins, labels=labels)

    # Calculating percentage of q>1 events for each AI bin and fractional_wood_pct
    q_greater_1 = (
        df[df[y_varname] > 1]
        .groupby([x2_new_varname, x1_new_varname])
        .agg(
            count_greater_1=(y_varname, "size"),  # Count the occurrences
            sum_weightfact_q_gt_1=(weight_by, "sum"),  # Sum the event_length
        )
        .reset_index()
    )

    total_counts = (
        df.groupby([x2_new_varname, x1_new_varname])
        .agg(
            total_count=(y_varname, "size"),  # Count the occurrences
            sum_weightfact_total=(weight_by, "sum"),  # Sum the event_length
        )
        .reset_index()
    )
    percentage_df = pd.merge(
        q_greater_1, total_counts, on=[x2_new_varname, x1_new_varname]
    )

    percentage_df["percentage_q_gt_1"] = (
        percentage_df["count_greater_1"] / percentage_df["total_count"]
    ) * 100
    percentage_df["percentage_q_le_1"] = 100 - percentage_df["percentage_q_gt_1"]
    percentage_df["percentage_q_le_1"] = percentage_df["percentage_q_le_1"].fillna(0)
    percentage_df["percentage_q_gt_1"] = percentage_df["percentage_q_gt_1"].fillna(0)

    # count percentage * days percentage
    # percentage_df["weighted_percentage_q_gt_1"] = percentage_df["percentage_q_gt_1"] * percentage_df["sum_event_length_q_gt_1"]/ percentage_df["sum_event_length_total"]
    # percentage_df["weighted_percentage_q_le_1"] = percentage_df["percentage_q_le_1"] * (percentage_df["sum_event_length_total"]-percentage_df["sum_event_length_q_gt_1"])/ percentage_df["sum_event_length_total"]

    # Percentage of count * days
    percentage_df["count_x_weight_q_gt_1"] = (
        percentage_df["count_greater_1"] * percentage_df["sum_weightfact_q_gt_1"]
    )
    percentage_df["count_x_weight_q_le_1"] = (
        percentage_df["total_count"] - percentage_df["count_greater_1"]
    ) * (percentage_df["sum_weightfact_total"] - percentage_df["sum_weightfact_q_gt_1"])
    percentage_df["weighted_percentage_q_gt_1"] = (
        percentage_df["count_x_weight_q_gt_1"]
        / (
            percentage_df["count_x_weight_q_gt_1"]
            + percentage_df["count_x_weight_q_le_1"]
        )
        * 100
    )
    percentage_df["weighted_percentage_q_le_1"] = (
        percentage_df["count_x_weight_q_le_1"]
        / (
            percentage_df["count_x_weight_q_gt_1"]
            + percentage_df["count_x_weight_q_le_1"]
        )
        * 100
    )
    return percentage_df


percentage_df = get_df_percentage_q(
    df=df_filt_q_conus,
    x1_varname="fractional_wood",
    x2_varname="AI",
    y_varname="q_q",
    weight_by="event_length",
)
# percentage_df_median = get_df_percentage_q(df=df_filt_q_conus_agg, x1_varname="fractional_wood_median", x2_varname="AI_median", y_varname="q_q_median", weight_by="event_length_median")
# %%
percentage_df.to_csv(os.path.join(fig_dir, f"sample_wood_stat.csv"))


# %%
# Plot the q<1 and q>1 proportion with aridity and fractional woody vegetation cover
def darken_hex_color(hex_color, darken_factor=0.7):
    # Convert hex to RGB
    rgb_color = mcolors.hex2color(hex_color)
    # Convert RGB to HSV
    hsv_color = mcolors.rgb_to_hsv(rgb_color)
    # Darken the color by reducing its V (Value) component
    hsv_color[2] *= darken_factor
    # Convert back to RGB, then to hex
    darkened_rgb_color = mcolors.hsv_to_rgb(hsv_color)
    return mcolors.to_hex(darkened_rgb_color)


def plot_grouped_stacked_bar(
    ax, df, x_column_to_plot, z_var, var_name, title_name, weighted=False
):

    # Determine unique groups and categories
    # Define the width of the bars and the space between groups
    bar_width = 0.2
    space_between_bars = 0.025
    space_between_groups = 0.2

    # Determine unique values for grouping
    # Aridity bins
    x_unique = df[x_column_to_plot].unique()[:-1]
    # Vegetation bins
    z_unique = df[z_var].unique()
    n_groups = len(z_unique)

    # Define original colors
    base_colors = ["#FFE268", "#22BBA9"]  # (q<1, q>1)
    min_darken_factor = 0.85

    # Setup for weighted or unweighted percentages
    if weighted:
        y_vars = ["weighted_percentage_q_le_1", "weighted_percentage_q_gt_1"]
    else:
        y_vars = ["percentage_q_le_1", "percentage_q_gt_1"]

    # Create the grouped and stacked bars
    for z_i, z in enumerate(z_unique):
        for x_i, x in enumerate(x_unique):

            # Darken colors for this group
            darken_factor = max(
                np.sqrt(np.sqrt(np.sqrt(1 - (x_i / len(x_unique))))), min_darken_factor
            )
            colors = [darken_hex_color(color, darken_factor) for color in base_colors]

            # Calculate the x position for each group
            group_offset = (bar_width + space_between_bars) * n_groups
            x_pos = (
                x_i * (group_offset + space_between_groups)
                + (bar_width + space_between_bars) * z_i
            )

            # Get the subset of df for this group
            subset = df[(df[x_column_to_plot] == x) & (df[z_var] == z)]

            # Get bottom values for stacked bars
            bottom_value = 0

            for i, (y_var, color) in enumerate(zip(y_vars, colors)):
                ax.bar(
                    x_pos,
                    subset[y_var].values[0],
                    bar_width,
                    bottom=bottom_value,
                    color=color,
                    edgecolor="white",
                    label=(
                        f'{z} - {y_var.split("_")[-1]}' if x_i == 0 and i == 0 else ""
                    ),
                )

                bottom_value += subset[y_var].values[0]

    # Set the x-ticks to the middle of the groups
    ax.set_xticks(
        [
            i * (group_offset + space_between_groups) + group_offset / 2
            for i in range(len(x_unique))
        ]
    )
    ax.set_xticklabels(x_unique, rotation=45)
    ax.set_xlabel(f"{var_dict["ai_bins"]['label']} {var_dict["ai_bins"]['unit']}")

    # Set the y-axis
    if weighted:
        ax.set_ylabel("Weighted proportion of\ndrydown events by duration (%)")
        ax.set_ylim([0, 15])
    else:
        ax.set_ylabel("Proportion of\ndrydown events (%)")
        ax.set_ylim([0, 40])

    # Set the second x-ticks
    # Replicate the z_var labels for the number of x_column_to_plot labels
    z_labels = np.tile(z_unique, len(x_unique))

    # Adjust the tick positions for the replicated z_var labels
    new_tick_positions = [i + bar_width / 2 for i in range(len(z_labels))]

    # Hide the original x-axis ticks and labels
    ax.tick_params(axis="x", which="both", length=0)

    # Create a secondary x-axis for the new labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_positions)
    ax2.set_xlabel(f"{var_dict[var_name]['label']} {var_dict[var_name]['unit']}")

    # Adjust the secondary x-axis to appear below the primary x-axis
    ax2.spines["top"].set_visible(False)
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 60))
    ax2.tick_params(axis="x", which="both", length=0)
    ax2.set_xticklabels(z_labels, rotation=45)

    # Set plot title and legend
    ax.set_title(title_name)


plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(7, 4))
plot_grouped_stacked_bar(
    ax=ax,
    df=percentage_df,
    x_column_to_plot="AI_binned2",
    z_var="fractional_wood_pct",
    var_name="rangeland_wood",
    title_name="",
    weighted=False,
)
plt.tight_layout()
plt.show()
save_figure(fig, fig_dir, "fracwood_q_unweighted", "pdf", 1200)

fig, ax = plt.subplots(figsize=(7, 4))
plot_grouped_stacked_bar(
    ax=ax,
    df=percentage_df,
    x_column_to_plot="AI_binned2",
    z_var="fractional_wood_pct",
    var_name="rangeland_wood",
    title_name="",
    weighted=True,
)
plt.tight_layout()
plt.show()
save_figure(fig, fig_dir, "fracwood_q_weighted", "pdf", 1200)

# %%
#############################
# Nonveg area for supplementals

# Get "other" land-use percent in vegetated area
df_filt_q_conus["nonveg_percent"] = df_filt_q_conus["barren_percent"] + (
    100 - df_filt_q_conus["totalrangeland_percent"]
)
percentage_df_nonveg20 = get_df_percentage_q(
    df=df_filt_q_conus[df_filt_q_conus["nonveg_percent"] < 20],
    x1_varname="nonveg_percent",
    x2_varname="AI",
    y_varname="q_q",
    weight_by="event_length",
    bins=[0, 5, 10, 15, 20],
    labels=["0-5%", "5-10%", "10-15%", "15-20%"],
)

fig, ax = plt.subplots(figsize=(7, 4))
plot_grouped_stacked_bar(
    ax=ax,
    df=percentage_df_nonveg20,
    x_column_to_plot="AI_binned2",
    z_var="nonveg_percent_pct",
    var_name="rangeland_other",
    title_name="",
    weighted=False,
)
plt.tight_layout()
plt.show()
save_figure(fig, fig_dir, "fracwood_q_nonveg_impact", "pdf", 1200)

# %%

# Just to get q legend texts ...

# Create a figure and axis
fig, ax = plt.subplots()

# Set limits and remove axis
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Plot text in the middle of the figure
ax.text(0.5, 0.5, r"$\hat{q}<1$", fontsize=20, ha="center", va="center")
ax.text(0.5, 0.3, r"$\hat{q}>1$", fontsize=20, ha="center", va="center")

# Show the plot
plt.show()
save_figure(fig, fig_dir, "fracwood_q_text", "pdf", 1200)
# %%
# %%
##########################################################################################
# Histogram with mean and median
###########################################################################################


def plot_histograms_with_mean_median(
    df, x_var, z_var, cmap=None, categories=None, colors=None
):
    if categories is None:
        # Get unique bins
        bins_in_range = df[z_var["column_name"]].unique()
        bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
        bins_sorted = sorted(bins_list, key=lambda x: x.left)
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / len(bins_sorted)) for i in range(len(bins_sorted))]
    else:
        bins_sorted = categories

    # Determine the number of rows needed for subplots based on the number of categories
    n_rows = len(bins_sorted)
    fig, axes = plt.subplots(n_rows, 1, figsize=(4, 3 * n_rows))

    if n_rows == 1:
        axes = [axes]  # Make it iterable even for a single category

    # For each row in the subset, calculate the loss for a range of theta values
    for i, (category, ax) in enumerate(zip(bins_sorted, axes)):
        subset = df[df[z_var["column_name"]] == category]

        # Determine bin edges based on bin interval
        bin_interval = 0.2
        min_edge = 0
        max_edge = 10
        bins = np.arange(min_edge, max_edge + bin_interval, bin_interval)

        # Plot histogram
        sns.histplot(
            subset[x_var["column_name"]],
            label="histogram",
            color=colors[i],
            bins=bins,
            kde=False,
            ax=ax,
        )

        # Calculate and plot mean and median lines
        mean_value = subset[x_var["column_name"]].mean()
        median_value = subset[x_var["column_name"]].median()
        ax.axvline(mean_value, color=colors[i], linestyle=":", lw=2, label="mean")
        ax.axvline(median_value, color=colors[i], linestyle="-", lw=2, label="median")

        # Creating a KDE (Kernel Density Estimation) of the df
        kde = gaussian_kde(subset[x_var["column_name"]])

        # Creating a range of values to evaluate the KDE
        kde_values = np.linspace(0, max(subset[x_var["column_name"]]), 1000)

        kde.set_bandwidth(bw_method=kde.factor / 3.0)

        # Evaluating the KDE
        kde_evaluated = kde(kde_values)

        # Finding the peak of the KDE
        peak_kde_value = kde_values[np.argmax(kde_evaluated)]

        # Plotting the KDE
        ax.plot(kde_values, kde_evaluated, color=colors[i])

        # Highlighting the peak of the KDE
        ax.axvline(
            x=peak_kde_value,
            color=colors[i],
            linestyle="--",
            linewidth=2.5,
            label="mode",
        )

        # Set titles and labels for each subplot
        ax.set_title(f"{z_var['label']}: {category}")
        ax.set_xlabel(f"{x_var['symbol']} {x_var['unit']}")
        ax.set_ylabel("Frequency\n(Number of drydown events)")

        ax.set_xlim(0, x_var["lim"][1] * 2)
        ax.legend()

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

    return fig, ax


# %% df_filt_allq: Including extremely small  q values as well
plt.rcParams.update({"font.size": 8})
fig_hist_q_veg, _ = plot_histograms_with_mean_median(
    df=df_filt_q,
    x_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)

fig_hist_q_veg.savefig(
    os.path.join(fig_dir, f"sup_hist_q_veg_allq.png"), dpi=600, bbox_inches="tight"
)
# %%
fig_hist_q_ai2, _ = plot_histograms_with_mean_median(
    df=df_filt_q, x_var=var_dict["q_q"], z_var=var_dict["ai_bins"], cmap=ai_cmap
)

fig_hist_q_ai2.savefig(
    os.path.join(fig_dir, f"sup_hist_q_ai_allq.png"), dpi=1200, bbox_inches="tight"
)

# %%
fig_hist_q_sand2, _ = plot_histograms_with_mean_median(
    df=df_filt_q, x_var=var_dict["q_q"], z_var=var_dict["sand_bins"], cmap=sand_cmap
)

fig_hist_q_sand2.savefig(
    os.path.join(fig_dir, f"sup_hist_q_sand_allq.png"), dpi=1200, bbox_inches="tight"
)


# %% #####################################################
# Statistical significance #############################
#####################################################
def stat_dist_test(df, x_var, z_var, cmap=None, categories=None, colors=None):
    if categories is None:
        # Get unique bins
        bins_in_range = df[z_var["column_name"]].unique()
        bins_list = [bin for bin in bins_in_range if pd.notna(bin)]
        bins_sorted = sorted(bins_list, key=lambda x: x.left)
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i / len(bins_sorted)) for i in range(len(bins_sorted))]
    else:
        bins_sorted = categories

    # Prepare DataFrame to store the p-values
    p_values_mw = pd.DataFrame(index=bins_sorted, columns=bins_sorted)
    p_values_ks = pd.DataFrame(index=bins_sorted, columns=bins_sorted)
    p_values_median = pd.DataFrame(index=bins_sorted, columns=bins_sorted)

    # For each row in the subset, calculate the loss for a range of theta values
    for i, category in enumerate(bins_sorted):
        subset = df[df[z_var["column_name"]] == category]

        # Calculate the statistical differences between each pair of categories
        for j, other_category in enumerate(bins_sorted):
            if i > j:
                other_subset = df[df[z_var["column_name"]] == other_category]
                _, p_mw = mannwhitneyu(
                    subset[x_var["column_name"]].values,
                    other_subset[x_var["column_name"]].values,
                    alternative="two-sided",
                )
                _, p_ks = ks_2samp(
                    subset[x_var["column_name"]].values,
                    other_subset[x_var["column_name"]].values,
                )
                stat, p_med, _, _ = median_test(
                    subset[x_var["column_name"]].values,
                    other_subset[x_var["column_name"]].values,
                )

                # p_values_mw.at[category, other_category] = p_mw
                p_values_mw.at[other_category, category] = p_mw
                # p_values_ks.at[category, other_category] = p_ks
                p_values_ks.at[other_category, category] = p_ks
                # p_values_median.at[category, other_category] = p_med
                p_values_median.at[other_category, category] = p_med

    # Outputting the p-values as a heatmap for visibility
    p_values_mw = p_values_mw.apply(pd.to_numeric, errors="coerce").fillna(np.nan)
    # p_values_ks = p_values_ks.apply(pd.to_numeric, errors="coerce").fillna(np.nan)
    p_values_median = p_values_median.apply(pd.to_numeric, errors="coerce").fillna(
        np.nan
    )

    # Create a custom colormap
    colors = ["#2b8cbe", "#a6bddb", "#ece7f2"]  # dark blue, blue, white
    boundaries = [0, 1.0e-2, 0.05, 1]  # values for the boundaries between colors
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors), clip=True)

    # Plotting the heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(
        p_values_mw,
        mask=p_values_mw.isnull(),
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        ax=axes[0],
    )
    axes[0].set_title("(a) Mann-Whitney U Test P-values")
    # sns.heatmap(
    #     p_values_ks,
    #     mask=p_values_ks.isnull(),
    #     annot=True,
    #     fmt=".2f",
    #     cmap=cmap,
    #     norm=norm,
    #     ax=axes[1],
    # )
    # axes[0, 1].set_title("(b) Kolmogorov-Smirnov Test P-values")
    sns.heatmap(
        p_values_median,
        mask=p_values_median.isnull(),
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        ax=axes[1],
    )
    axes[1].set_title("(b) Mood’s Median Test P-values")

    for ax in [axes[0], axes[1]]:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=40, ha="right"
        )  # Adjusting the tick position and angle

    plt.tight_layout()
    # print(p_values_mw)
    # print(p_values_ks)
    print(p_values_median)

    return fig, _


plt.rcParams.update({"font.size": 12})
fig_stat_test_veg, _ = stat_dist_test(
    df=df_filt_q,
    x_var=var_dict["q_q"],
    z_var=var_dict["veg_class"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)

_, p_mw = mannwhitneyu(
    df_filt_q[var_dict["q_q"]["column_name"]].values,
    df_filt_q[var_dict["AI"]["column_name"]].values,
    alternative="two-sided",
)
print(f"M-W U test: {p_mw:.3f}")
_, p_ks = ks_2samp(
    df_filt_q[var_dict["q_q"]["column_name"]].values,
    df_filt_q[var_dict["AI"]["column_name"]].values,
)
print(f"K-S test: {p_ks:.3f}")
stat, p_med, _, _ = median_test(
    df_filt_q[var_dict["q_q"]["column_name"]].values,
    df_filt_q[var_dict["AI"]["column_name"]].values,
)
print(f"Median test: {p_med:.3f}")

plt.rcParams.update({"font.size": 12})
fig_stat_test_veg, _ = stat_dist_test(
    df=df_filt_q_agg,
    x_var=var_dict["q_q_median"],
    z_var=var_dict["veg_class_mode"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
)

if save:
    save_figure(fig_stat_test_veg, fig_dir, f"sup_veg_statsignificance", "pdf", 1200)


# %%
# %% #####################################################
# Statistical significance #############################
#####################################################


# Calculate the point density
def plot_contour(ax, df, x_var, y_var, cmap, title):

    xdata = df[var_dict[x_var]["column_name"]].values
    ydata = df[var_dict[y_var]["column_name"]].values
    xy = np.vstack([xdata, ydata])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = xdata[idx], ydata[idx], z[idx]

    # Create grid to interpolate data
    grid_x, grid_y = np.mgrid[
        min(x) : max(x) : 100j, min(y) : max(y) : 100j
    ]  # 100j specifies 100 points in each dimension

    # Interpolate z values on grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method="cubic")

    CS = ax.contour(
        grid_x, grid_y, grid_z, levels=15, cmap=cmap
    )  # Adjust number of levels as needed
    ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
    ax.set_title(title, loc="left")

    # Display correlation and p-value
    correlation, p_value = spearmanr(x, y)
    if p_value < 1.0e-2:
        extension = "*"
    else:
        extension = ""
    ax.text(
        0.95,
        0.05,
        rf"Spearman's $\rho$ = {correlation:.2f} {extension}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    )

    ax.set_xlim(var_dict[x_var]["lim"][0], var_dict[x_var]["lim"][1])
    ax.set_ylim(var_dict[y_var]["lim"][0], var_dict[y_var]["lim"][1])
    ax.set_xlabel(f'{var_dict[x_var]["symbol"]} {var_dict[x_var]["unit"]}')
    ax.set_ylabel(f'{var_dict[y_var]["symbol"]} {var_dict[y_var]["unit"]}')


contour_cmap = "PuBu"

# %%
############################################
# 4-grids with contour - Aridity Index - median
############################################
plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

var_key = "AI_median"
plot_contour(
    ax=axs[0],
    df=df_filt_q_agg,
    x_var=var_key,
    y_var="q_q_median",
    cmap=contour_cmap,
    title="(a)",
)
plot_contour(
    ax=axs[1],
    df=df_filt_q_agg,
    x_var=var_key,
    y_var="q_ETmax_median",
    cmap=contour_cmap,
    title="(b)",
)
plot_contour(
    ax=axs[2],
    df=df_filt_q_agg,
    x_var=var_key,
    y_var="q_theta_star_median",
    cmap=contour_cmap,
    title="(c)",
)

plt.tight_layout()
plt.show()
if save:
    save_figure(fig, fig_dir, f"sup_lossfnc_ai_median_by_pixel_w_contour", "png", 1200)
    save_figure(fig, fig_dir, f"sup_lossfnc_ai_median_by_pixel_w_contour", "pdf", 1200)

# %%
############################################
# 4-grids with contour - Sand fraction - median
############################################

del fig, axs
plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

var_key = "sand_fraction_median"
plot_contour(
    ax=axs[0],
    df=df_filt_q_agg.dropna(subset=["sand_fraction_median"]),
    x_var=var_key,
    y_var="q_q_median",
    cmap=contour_cmap,
    title="(d)",
)
plot_contour(
    ax=axs[1],
    df=df_filt_q_agg.dropna(subset=["sand_fraction_median"]),
    x_var=var_key,
    y_var="q_ETmax_median",
    cmap=contour_cmap,
    title="(e)",
)
plot_contour(
    ax=axs[2],
    df=df_filt_q_agg.dropna(subset=["sand_fraction_median"]),
    x_var=var_key,
    y_var="q_theta_star_median",
    cmap=contour_cmap,
    title="(f)",
)

plt.tight_layout()
plt.show()
if save:
    save_figure(
        fig, fig_dir, f"sup_lossfnc_sand_median_by_pixel_w_contour", "png", 1200
    )
    save_figure(
        fig, fig_dir, f"sup_lossfnc_sand_median_by_pixel_w_contour", "pdf", 1200
    )

# %% Get legends of loss function
del fig, axs
plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
plot_loss_func(
    ax=axs,
    df=df_filt_q_agg,
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    plot_legend=True,
    title="(a)",
    median_by_pixel=True,
)
save_figure(fig, fig_dir, f"sup_lossfnc_ai_legend", "pdf", 1200)

plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
plot_loss_func(
    ax=axs,
    df=df_filt_q_agg,
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    plot_legend=True,
    title="(a)",
    median_by_pixel=True,
)
save_figure(fig, fig_dir, f"sup_lossfnc_sand_legend", "pdf", 1200)

# %%
# #################################################################
# Relationship between q and the length of the drydown events
#################################################################

plt.rcParams.update({"font.size": 14})


# Calculate the point density
def plot_eventlength_vs_q(df, x_var, y_var, cmap):

    xdata = df[var_dict[x_var]["column_name"]].values
    ydata = df[var_dict[y_var]["column_name"]].values
    xy = np.vstack([xdata, ydata])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = xdata[idx], ydata[idx], z[idx]

    fig, ax = plt.subplots(figsize=(6, 3))
    scatter = ax.scatter(x, y, c=z, s=50, cmap=cmap)

    ax.set_xlim([4, 30])
    ax.set_ylim([0, 10])
    ax.set_xlabel("Event duration (days)")
    ax.set_ylabel(r"$\hat{q}$ (-)")

    # Create colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Density")
    return fig, ax


fig, ax = plot_eventlength_vs_q(
    df_filt_q_conus[df_filt_q_conus["totalrangeland_percent"] > 80],
    "event_length",
    "q_q",
    "PuBu",
)
plt.show()
# save_figure(fig, fig_dir, "sup_evenglength_vs_q", "pdf", 1200)
save_figure(fig, fig_dir, "sup_evenglength_vs_q", "png", 1200)
# %%
# Reset stdout to print to the console
sys.stdout = original_stdout
f.close()  # Manually close the file

# Additional code here will print to the console
print(f"All operations have been logged at {fig_dir}")
print("Plotting complete")
# %%


###############################################################################
###############################################################################
###############################################################################
# Other sandbox
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# %%
# %%
# # Calculate the point density
# def plot_scatter_with_density(df, x_var, y_var, cmap):

#     xdata = df[var_dict[x_var]["column_name"]].values
#     ydata = df[var_dict[y_var]["column_name"]].values
#     xy = np.vstack([xdata, ydata])
#     z = gaussian_kde(xy)(xy)

#     # Sort the points by density, so that the densest points are plotted last
#     idx = z.argsort()
#     x, y, z = xdata[idx], ydata[idx], z[idx]

#     fig, ax = plt.subplots()
#     ax.scatter(x, y, c=z, s=50, cmap=cmap)
#     ax.set_title("Density Plot with Scatter")

#     ax.set_xlim(f'{var_dict[x_var]["symbol"]} {var_dict[x_var]["unit"]}')
#     ax.set_ylabel(f'{var_dict[y_var]["symbol"]} {var_dict[y_var]["unit"]}')
#     ax.set_xlabel(f'{var_dict[x_var]["symbol"]} {var_dict[x_var]["unit"]}')
#     ax.set_ylabel(f'{var_dict[y_var]["symbol"]} {var_dict[y_var]["unit"]}')
#     plt.show()
#     return fig, ax
# def plot_eventlength_vs_q(df):
#     fig, ax = plt.subplots(figsize=(6, 4))  # Create a figure and an axes object

#     # Use seaborn's regplot to plot data with a regression line (trendline) and confidence interval
#     sns.regplot(
#         x="event_length", y="q_q", data=df, ax=ax, ci=95
#     )  # ci parameter controls the confidence interval
#     ax.set_ylabel(r"$q$ (-)")
#     ax.set_xlabel("Event duration(days)")

#     return fig, ax  # Return figure and axes object for further manipulation or saving


# # Define the plotting function
# def plot_eventlength_vs_q(df):
#     fig, ax = plt.subplots(figsize=(7, 4))  # Create a figure and an axes object
#     scatter = ax.scatter(
#         df["event_length"], df["q_q"], marker=".", alpha=0.3
#     )  # Plot data
#     ax.set_ylabel(r"$q$")
#     ax.set_xlabel("Length of the event [days]")
#     # ax.set_xlim([0, 20])  # Optional: uncomment to set x-axis limits
#     return fig, ax  # Return figure and axes object for further manipulation or saving
# fig, ax = plot_eventlength_vs_q(df_filt_q_conus)
# plt.show()


# %%


# def plot_eventlength_hist(df):
#     min_value = df["event_length"].min()
#     max_value = df["event_length"].max()
#     bin_width = 1
#     n_bins = (
#         int((max_value - min_value) / bin_width) + 1
#     )  # Adding 1 to include the max value

#     hist = df["event_length"].hist(bins=n_bins)
#     hist.set_xlabel("Length of the event [days]")
#     hist.set_ylabel("Frequency")
#     # hist.set_xlim([0, 20])


# # plot_eventlength_hist(df_filt_q)
# plot_eventlength_hist(df_filt_q_conus[~pd.isna(df_filt_q_conus["barren_percent"])])


# %%

# %%
# # %%
# def using_mpl_scatter_density(df, x_var, y_var, cmap):
#     fig, ax = plt.subplots()
#     xdata = df[x_var['column_name']].values
#     ydata = df[y_var['column_name']].values
#     ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
#     ax.scatter_density(xdata, ydata, cmap=cmap)
#     ax.set_xlim(x_var['lim'][0], x_var['lim'][1])
#     ax.set_ylim(y_var['lim'][0], y_var['lim'][1])
#     # fig.colorbar(ax, label="Number of points per pixel")

# # %%
# using_mpl_scatter_density(df=df_filt_q, x_var=var_dict["AI"], y_var=var_dict["q_q"], cmap="Blues")
# using_mpl_scatter_density(df=df_filt_q_agg, x_var=var_dict["AI_median"], y_var=var_dict["q_q_median"], cmap="Blues")

# # %%
# using_mpl_scatter_density(df=df_filt_q.dropna(subset=["sand_fraction"]), x_var=var_dict["sand_fraction"], y_var=var_dict["q_q"], cmap=sand_cmap)
# using_mpl_scatter_density(df=df_filt_q_agg.dropna(subset=["sand_fraction"]), x_var=var_dict["sand_fraction_median"], y_var=var_dict["q_q_median"], cmap=sand_cmap)

# # %%
# # %%
# def plot_grouped_stacked_bar_uni(ax, df, z_var, var_name, title_name, weighted=False):

#     # Determine unique groups and categories
#     # Define the width of the bars and the space between groups
#     bar_width = 0.2
#     space_between_bars = 0.025
#     space_between_groups = 0.2

#     # Determine unique values for grouping

#     # Vegetation bins
#     z_unique = df[z_var].unique()
#     n_groups = len(z_unique)
#     # exclude = df["AI_binned2"].unique([:1])

#     # exclude = df["AI_binned2"].unique()[-1]

#     # Define original colors
#     base_colors = ["#FFE268", "#22BBA9"]  # (q<1, q>1)
#     min_darken_factor = 0.85

#     # Setup for weighted or unweighted percentages
#     if weighted:
#         y_vars = ["weighted_percentage_q_le_1", "weighted_percentage_q_gt_1"]
#     else:
#         y_vars = ["percentage_q_le_1", "percentage_q_gt_1"]

#     # Create the grouped and stacked bars
#     for z_i, z in enumerate(z_unique):

#         # Get the subset of df for this group
#         subset = df[(df[z_var] == z)]  # &(df["AI_binned2"] != exclude)]
#         # print(subset.total_count.sum())
#         # Check the sample size before plotting
#         if (
#             subset.total_count.sum() < 100
#         ):  # If the number of samples in the group is less than 10, skip plotting
#             continue

#         # Get bottom values for stacked bars
#         bottom_value = 0
#         # Calculate the x position for each group
#         group_offset = (bar_width + space_between_bars) * n_groups
#         x_pos = (group_offset + space_between_groups) + (
#             bar_width + space_between_bars
#         ) * z_i

#         for i, (y_var, color) in enumerate(zip(y_vars, base_colors)):
#             ax.bar(
#                 x_pos,
#                 subset[y_var].values[0],
#                 bar_width,
#                 bottom=bottom_value,
#                 color=color,
#                 edgecolor="white",
#             )

#             bottom_value += subset[y_var].values[0]

#     # Set the y-axis
#     if weighted:
#         ax.set_ylabel("Weighted proportion of\ndrydown events\nby event length (%)")
#         ax.set_ylim([0, 20])
#     else:
#         ax.set_ylabel("Proportion of\ndrydown events (%)")
#         ax.set_ylim([0, 40])

#     # Set the second x-ticks
#     # Replicate the z_var labels for the number of x_column_to_plot labels
#     z_labels = z_unique

#     # Adjust the tick positions for the replicated z_var labels
#     new_tick_positions = [i + bar_width / 2 for i in range(len(z_labels))]

#     # Hide the original x-axis ticks and labels
#     ax.tick_params(axis="x", which="both", length=0)

#     # Create a secondary x-axis for the new labels
#     ax2 = ax.twiny()
#     ax2.set_xlim(ax.get_xlim())
#     ax2.set_xticks(new_tick_positions)
#     ax2.set_xlabel(f"{var_dict[var_name]['label']} {var_dict[var_name]['unit']}")

#     # Adjust the secondary x-axis to appear below the primary x-axis
#     ax2.spines["top"].set_visible(False)
#     ax2.xaxis.set_ticks_position("bottom")
#     ax2.xaxis.set_label_position("bottom")
#     ax2.spines["bottom"].set_position(("outward", 60))
#     ax2.tick_params(axis="x", which="both", length=0)
#     ax2.set_xticklabels(z_labels, rotation=45)

#     # Set plot title and legend
#     ax.set_title(title_name)


# percentage_df_10 = get_df_percentage_q(
#     df_filt_q_conus,
#     "fractional_wood",
#     "AI",
#     "q_q",
#     "event_length",
#     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#     [
#         "0-10%",
#         "10-20%",
#         "20-30%",
#         "30-40%",
#         "40-50%",
#         "50-60%",
#         "60-70%",
#         "70-80%",
#         "80-90%",
#         "90-100%",
#     ],
# )
# percentage_df_10 = get_df_percentage_q(
#     df_filt_q_conus,
#     "fractional_wood",
#     "AI",
#     "q_q",
#     "event_length",
#     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
#     [
#         "0-10%",
#         "10-20%",
#         "20-30%",
#         "30-40%",
#         "40-50%",
#         "50-60%",
#         "60-70%",
#         "70-80%",
#         "80-90%",
#         "90-100%",
#     ],
# )
# percentage_df_5 = get_df_percentage_q(
#     df_filt_q_conus,
#     "fractional_wood",
#     "AI",
#     "q_q",
#     "event_length",
#     [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
#     [
#         "0-5%",
#         "5-10%",
#         "10-15%",
#         "15-20%",
#         "20-25%",
#         "25-30%",
#         "30-35%",
#         "35-40%",
#         "40-45%",
#         "45-50%",
#         "50-55%",
#         "55-60%",
#         "60-65%",
#         "65-70%",
#         "70-75%",
#         "75-80%",
#         "80-85%",
#         "85-90%",
#         "90-95%",
#         "95-100%",
#     ],
# )

# fig, ax = plt.subplots(figsize=(4, 4))
# plot_grouped_stacked_bar_uni(
#     ax=ax,
#     df=percentage_df_10,
#     z_var="fractional_wood_pct",
#     var_name="rangeland_wood",
#     title_name="",
#     weighted=True,
# )
# plt.tight_layout()
# if save:
#     save_figure(fig, fig_dir, "fracwood_q_weighted", "pdf", 1200)

# fig, ax = plt.subplots(figsize=(4, 4))
# plot_grouped_stacked_bar_uni(
#     ax=ax,
#     df=percentage_df_10,
#     z_var="fractional_wood_pct",
#     var_name="rangeland_wood",
#     title_name="",
#     weighted=False,
# )
# plt.tight_layout()
# if save:
#     save_figure(fig, fig_dir, "fracwood_q_unweighted", "pdf", 1200)

# %%


# # %%
# # Plotting
# def plot_scatter(df, x_var, y_var, cmap="YlGn"):
#     fig, ax = plt.subplots(figsize=(5, 4))
#     # # Setting up the discrete colormap
#     cmap=plt.get_cmap(cmap)
#     x_bin_interval = (x_var["lim"][1] - x_var["lim"][0])/5
#     norm = plt.Normalize(vmin=x_var["lim"][0]-x_bin_interval, vmax=x_var["lim"][1])
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])

#     print(x_bin_interval)
#     print(np.arange(x_var["lim"][0], x_var["lim"][1], x_bin_interval))
#     # Adding regression lines per 5-year segments
#     for start in np.arange(x_var["lim"][0], x_var["lim"][1], x_bin_interval):

#         # Get df subset
#         subset = df[(df[x_var["column_name"]] >= start) & (df[x_var["column_name"]] < start + x_bin_interval)]

#         # Get color
#         midpoint = start + 2.5  # Midpoint for color indexing
#         color = cmap(norm(midpoint))
#         dark_color = [x * 0.8 for x in color[:3]] + [1]

#         # Plot trendline and scatter
#         sns.scatterplot(x=x_var["column_name"], y=y_var["column_name"], df=subset, alpha=0.3, color=color, ax=ax)
#         sns.regplot(x=x_var["column_name"], y=y_var["column_name"], df=subset, scatter=False,  color=dark_color, ax=ax)

#     for line in ax.get_lines():
#         line.set_linestyle('--')

#     # Adding a trend line
#     sns.regplot(x=x_var["column_name"], y=y_var["column_name"], df=df, scatter=False, color='black', ax=ax)

#     # Test the significance of the relationship
#     df_clean = df.dropna(subset=[x_var["column_name"], y_var["column_name"]])

#     # Compute Spearman correlation
#     correlation, p_value = spearmanr(df_clean[x_var["column_name"]].values, df_clean[y_var["column_name"]].values)
#     print("Spearman correlation")
#     print("Correlation:", correlation)
#     print("P-value:", p_value)

#     # Regression analysis
#     x = statsm.add_constant(df_clean[x_var["column_name"]].values)
#     model = statsm.OLS(df_clean[y_var["column_name"]].values, x)
#     results = model.fit()
#     print(results.summary())

#     # Enhancing the plot
#     plt.title('')
#     plt.xlabel(f'{x_var["label"]} {x_var["symbol"]} {x_var["unit"]}')
#     plt.ylabel(f'{y_var["label"]} {y_var["symbol"]} {y_var["unit"]}')
#     # plt.ylim([y_var["lim"][0], y_var["lim"][1]])

#     plt.show()
# # %%
# plt.rcParams.update({"font.size": 18})
# plot_scatter(df=df_filt_q_conus, x_var=var_dict["rangeland_wood"], y_var=var_dict["q_q"], cmap="YlGn")
# plot_scatter(df=df_filt_q_conus_agg, x_var=var_dict["rangeland_wood_median"], y_var=var_dict["q_q_median"], cmap="YlGn")
# plot_scatter(df=df_filt_q_conus_agg, x_var=var_dict["rangeland_wood_median"], y_var=var_dict["q_q_var"], cmap="YlGn")
# # %%
# plot_scatter(df=df_filt_q, x_var=var_dict["AI"], y_var=var_dict["q_q"], cmap=ai_cmap)
# plot_scatter(df=df_filt_q_agg, x_var=var_dict["AI_median"], y_var=var_dict["q_q_median"], cmap=ai_cmap)
# # TODO: debug this. only one color is showing up ...

# # %%

# # %%
# plot_scatter(df=df_filt_q, x_var=var_dict["PET"], y_var=var_dict["q_q"], cmap=ai_cmap)
# # plot_scatter(df=df_filt_q_agg, x_var=var_dict["PET"], y_var=var_dict["q_q"], cmap=ai_cmap)

# # %%
# df_filt_q.columns
# # %%

# # # Change to percentage (TODO: fix this in the df management)
# # df_filt_q_conus["fractional_wood"] = df_filt_q_conus["fractional_wood"] * 100
# # df_filt_q_conus["fractional_herb"] = df_filt_q_conus["fractional_herb"] * 100

# # Print some statistics
# print(f"Total number of drydown event with successful q fits: {len(df_filt_q)}")
# print(
#     f"Total number of drydown event with successful q fits & within CONUS: {sum(~pd.isna(df_filt_q_conus['fractional_wood']))}"
# )
# print(f"{sum(~pd.isna(df_filt_q_conus['fractional_wood']))/len(df_filt_q)*100:.2f}%")
# # %%
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

# # %% Loss function parameter by vegetaiton and AI, supplemental (support Figure 4)
# def wrap_at_space(text, max_width):
#     parts = text.split(" ")
#     wrapped_parts = [wrap(part, max_width) for part in parts]
#     return "\n".join([" ".join(wrapped_part) for wrapped_part in wrapped_parts])


# def plot_box_ai_veg(df):
#     plt.rcParams.update({"font.size": 26})  # Adjust the font size as needed

#     fig, ax = plt.subplots(figsize=(20, 8))
#     for i, category in enumerate(vegetation_color_dict.keys()):
#         subset = df[df["name"] == category]
#         sns.boxplot(
#             x="name",
#             y="AI",
#             df=subset,
#             color=vegetation_color_dict[category],
#             ax=ax,
#             linewidth=2,
#         )

#     # ax = sns.violinplot(x='abbreviation', y='q_q', df=filtered_df, order=vegetation_orders, palette=palette_dict) # boxprops=dict(facecolor='lightgray'),
#     max_label_width = 20
#     ax.set_xticklabels(
#         [
#             wrap_at_space(label.get_text(), max_label_width)
#             for label in ax.get_xticklabels()
#         ]
#     )
#     plt.setp(ax.get_xticklabels(), rotation=45)

#     # ax.set_xticklabels([textwrap.fill(t.get_text(), 10) for t in ax.get_xticklabels()])
#     ax.set_ylabel("Aridity index [MAP/MAE]")
#     ax.set_xlabel("IGBP Landcover Class")
#     ax.set_ylim(0, 2.0)
#     ax.set_title("(a)", loc="left")
#     plt.tight_layout()

#     return fig, ax


# fig_box_ai_veg, _ = plot_box_ai_veg(df_filt_q)
# fig_box_ai_veg.savefig(
#     os.path.join(fig_dir, f"sup_box_ai_veg.png"), dpi=1200, bbox_inches="tight"
# )

# # # %%
# # # # %% Vegetation vs AI Boxplot
# # fig_ai_vs_veg, axs = plt.subplots(1, 2, figsize=(12, 6))
# # plot_scatter_with_errorbar_categorical(
# #     ax=axs[0],
# #     df=df_filt_q,
# #     x_var=var_dict["ai"],
# #     y_var=var_dict["theta_star"],
# #     z_var=var_dict["veg_class"],
# #     categories=vegetation_color_dict.keys(),
# #     colors=list(vegetation_color_dict.values()),
# #     title="(b)",
# #     quantile=25,
# #     plot_logscale=False,
# #     plot_legend=False,
# # )

# # plot_scatter_with_errorbar_categorical(
# #     ax=axs[1],
# #     df=df_filt_q,
# #     x_var=var_dict["ai"],
# #     y_var=var_dict["q_ETmax"],
# #     z_var=var_dict["veg_class"],
# #     categories=vegetation_color_dict.keys(),
# #     colors=list(vegetation_color_dict.values()),
# #     title="(c)",
# #     quantile=25,
# #     plot_logscale=True,
# #     plot_legend=False,
# # )

# # fig_ai_vs_veg.tight_layout()
# # fig_ai_vs_veg.savefig(
# #     os.path.join(fig_dir, f"sup_ai_vs_veg.png"), dpi=1200, bbox_inches="tight"
# # )


# # %%
# ###########################################################################
# ###########################################################################
# ############################################################################
# # Other plots (sandbox)
# ###########################################################################
# ###########################################################################
# ###########################################################################


# # %%
# ############################################################################
# # Box plots (might go supplemental)
# ###########################################################################
# def plot_boxplots(df, x_var, y_var):
#     plt.rcParams.update({"font.size": 12})
#     fig, ax = plt.subplots(figsize=(6, 4))

#     sns.boxplot(
#         x=x_var["column_name"],
#         y=y_var["column_name"],
#         df=df,
#         boxprops=dict(facecolor="lightgray"),
#         ax=ax,
#     )

#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     ax.set_xlabel(f'{x_var["label"]} {x_var["unit"]}')
#     ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
#     ax.set_ylim(y_var["lim"][0], y_var["lim"][1])
#     fig.tight_layout()

#     return fig, ax


# # %% sand
# fig_box_sand, _ = plot_boxplots(df_filt_q, var_dict["sand_bins"], var_dict["q_q"])
# fig_box_sand.savefig(
#     os.path.join(fig_dir, f"box_sand.png"), dpi=600, bbox_inches="tight"
# )
# # %% Aridity index
# fig_box_ai, _ = plot_boxplots(df_filt_q, var_dict["ai_bins"], var_dict["q_q"])
# fig_box_ai.savefig(os.path.join(fig_dir, f"box_ai.png"), dpi=600, bbox_inches="tight")


# # %% Vegatation
# def wrap_at_space(text, max_width):
#     parts = text.split(" ")
#     wrapped_parts = [wrap(part, max_width) for part in parts]
#     return "\n".join([" ".join(wrapped_part) for wrapped_part in wrapped_parts])


# def plot_boxplots_categorical(df, x_var, y_var, categories, colors):
#     # Create the figure and axes
#     fig, ax = plt.subplots(figsize=(8, 4))

#     # Plot the boxplot with specified colors and increased alpha
#     sns.boxplot(
#         x=x_var["column_name"],
#         y=y_var["column_name"],
#         df=df,
#         # hue=x_var['column_name'],
#         legend=False,
#         order=categories,
#         palette=colors,
#         ax=ax,
#     )

#     for patch in ax.artists:
#         r, g, b, a = patch.get_facecolor()
#         patch.set_facecolor(mcolors.to_rgba((r, g, b), alpha=0.5))

#     # Optionally, adjust layout
#     plt.tight_layout()
#     ax.set_xlabel(f'{x_var["label"]}')
#     max_label_width = 20
#     ax.set_xticklabels(
#         [
#             wrap_at_space(label.get_text(), max_label_width)
#             for label in ax.get_xticklabels()
#         ]
#     )
#     plt.setp(ax.get_xticklabels(), rotation=45)
#     ax.set_ylabel(f'{y_var["label"]} {y_var["unit"]}')
#     # Show the plot
#     ax.set_ylim(y_var["lim"][0], y_var["lim"][1] * 3)
#     plt.tight_layout()
#     plt.show()

#     return fig, ax


# # %%
# fig_box_veg, _ = plot_boxplots_categorical(
#     df_filt_q,
#     var_dict["veg_class"],
#     var_dict["q_q"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
# )
# fig_box_veg.savefig(os.path.join(fig_dir, f"box_veg.png"), dpi=600, bbox_inches="tight")


# # %%
# def plot_hist_diffR2(df, var_key):
#     plt.rcParams.update({"font.size": 30})
#     fig, ax = plt.subplots(figsize=(5.5, 5))

#     # Create the histogram with a bin width of 1
#     sns.histplot(
#         df[var_key], binwidth=0.005, color="#2c7fb8", fill=False, linewidth=3, ax=ax
#     )

#     # Setting the x limit
#     ax.set_xlim(-0.1, 0.1)

#     # Adding title and labels
#     # ax.set_title("Histogram of diffR2 values")
#     ax.set_xlabel(r"diffR2")
#     ax.set_ylabel("Frequency")

#     return fig, ax


# plot_hist_diffR2(df=df_filt_q_and_exp, var_key="diff_R2")

# # fig_thetastar_vs_et_ai.savefig(
# #     os.path.join(fig_dir, f"thetastar_vs_et_ai.png"), dpi=600, bbox_inches="tight"
# # )


# # %%
# pixel_counts = (
#     df_filt_q_and_exp.groupby(["EASE_row_index", "EASE_column_index"])
#     .size()
#     .reset_index(name="count")
# )
# plt.hist(
#     pixel_counts["count"],
#     bins=range(min(pixel_counts["count"]), max(pixel_counts["count"]) + 2, 1),
# )
# pixel_counts["count"].median()


# # # %% Ridgeplot for poster
# # def plot_ridgeplot(df, x_var, z_var, categories, colors):
# #     # # Create a figure
# #     # fig, ax = plt.subplots(figsize=(10, 10))

# #     # Create a FacetGrid varying by the categorical variable, using the order and palette defined
# #     g = sns.FacetGrid(
# #         df,
# #         row=z_var["column_name"],
# #         hue=z_var["column_name"],
# #         aspect=2.5,
# #         height=1.5,
# #         palette=colors,
# #         row_order=categories,
# #     )
# #     # https://stackoverflow.com/questions/45911709/limit-the-range-of-x-in-seaborn-distplot-kde-estimation

# #     # Map the kdeplot for the variable of interest across the FacetGrid
# #     def plot_kde_and_lines(x, color, label):
# #         ax = plt.gca()  # Get current axis
# #         sns.kdeplot(
# #             x,
# #             bw_adjust=0.1,
# #             clip_on=False,
# #             fill=True,
# #             alpha=0.5,
# #             clip=[0, 5],
# #             linewidth=0,
# #             color=color,
# #             ax=ax,
# #         )
# #         sns.kdeplot(
# #             x,
# #             bw_adjust=0.1,
# #             clip_on=False,
# #             clip=[0, 5],
# #             linewidth=2.5,
# #             color="w",
# #             ax=ax,
# #         )
# #         # Median
# #         median_value = x.median()
# #         ax.axvline(median_value, color=color, linestyle=":", lw=2, label="Median")
# #         # Mode (using KDE peak as a proxy)
# #         kde = gaussian_kde(x, bw_method=0.1)
# #         kde_values = np.linspace(x.min(), x.max(), 1000)
# #         mode_value = kde_values[np.argmax(kde(kde_values))]
# #         ax.axvline(mode_value, color=color, linestyle="--", lw=2, label="Mode")

# #     #
# #     g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# #     # Map the custom plotting function
# #     g.map(plot_kde_and_lines, x_var["column_name"])

# #     # Set the subplots to overlap
# #     g.fig.subplots_adjust(hspace=-2)

# #     # Add a horizontal line for each plot
# #     g.map(plt.axhline, y=0, lw=2, clip_on=False)

# #     # Define and use a simple function to label the plot in axes coordinates
# #     def label(x, color, label):
# #         ax = plt.gca()
# #         ax.text(
# #             0,
# #             0.2,
# #             label,
# #             fontweight="bold",
# #             color=color,
# #             ha="left",
# #             va="center",
# #             transform=ax.transAxes,
# #             size=16,
# #         )

# #     g.map(label, x_var["column_name"])

# #     # Remove axes details that don't play well with overlap
# #     g.set_titles("")
# #     g.set(yticks=[], ylabel="", xlabel=r"$q$ [-]")
# #     g.despine(bottom=True, left=True)

# #     # Adjust the layout
# #     plt.tight_layout()
# #     plt.show()

# #     return g


# # vegetation_color_dict_limit = {
# #     "Open shrublands": "#C99728",
# #     "Grasslands": "#13BFB2",
# #     "Savannas": "#92BA31",
# #     "Woody savannas": "#4C6903",
# #     "Croplands": "#F7C906",
# # }

# # fig_ridge_veg = plot_ridgeplot(
# #     df=df_filt_q,
# #     x_var=var_dict["q_q"],
# #     z_var=var_dict["veg_class"],
# #     categories=vegetation_color_dict_limit.keys(),
# #     colors=list(vegetation_color_dict_limit.values()),
# # )

# # fig_ridge_veg.savefig(
# #     os.path.join(fig_dir, f"sup_hist_ridge_veg.pdf"), dpi=1200, bbox_inches="tight"
# # )

# # fig_ridge_veg.savefig(
# #     os.path.join(fig_dir, f"sup_hist_ridge_veg.png"), dpi=1200, bbox_inches="tight"
# # )


# # %%


# # %%
# def plot_q_ai_wood_scatter(df):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     sc=ax.scatter(df["fractional_wood"], df["q_q"], c=df["AI"], cmap="RdBu", alpha=0.3)
#     ax.set_ylim(0,1.5)
#     ax.set_xlabel(var_dict["rangeland_wood"]["label"]+" "+var_dict["rangeland_wood"]["unit"])
#     ax.set_ylabel(var_dict["q_q"]["label"]+" "+var_dict["q_q"]["symbol"])
#     # Create a colorbar with the scatter plot's color mapping
#     cbar = plt.colorbar(sc, ax=ax)
#     cbar.set_label('Aridity Index (MAP/MAE')

# plot_q_ai_wood_scatter(df_filt_q_conus)

#     # %%
# # ##########################################################
# # # Similar plot but in scatter plot format
# # ##########################################################

# # Get unique AI_binned2 values and assign colors
# cmap = plt.get_cmap('RdBu')
# ai_bins_list = percentage_df['AI_binned2'].unique()
# colors = cmap(np.linspace(0, 1, len(ai_bins_list)))
# color_map = dict(zip(ai_bins_list, colors))

# plt.rcParams.update({"font.size": 16})
# fig, ax = plt.subplots(figsize=(8, 5))
# # for (ai_bin, group) in percentage_df.groupby('AI_binned2'):
# #     ax.plot(group['fractional_wood_pct'], group['percentage_q_gt_1'], label=ai_bin, alpha=0.7, marker='o', color=color_map[ai_bin])
# for (ai_bin, group) in percentage_df.groupby('AI_binned2'):
#     ax.plot(group['fractional_wood_pct'], group['percentage_q_gt_1'], label=ai_bin, alpha=0.7, marker='o', color=color_map[ai_bin])

# ax.set_xlabel('Fractional Wood Coverage (%)')
# ax.set_ylabel('Proportion of events with\n'+r'$q>1$'+'(convex non-linearity) (%)')
# ax.legend(title='Aridity Index\n[MAP/MAE]', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
# ax.set_ylim([60, 100])  # Adjusting y-axis limits to 0-100% for percentage
# plt.xticks(rotation=45)
# fig.tight_layout()


# %%
#############################################################
# Get the Barren + Litter + Other percentage
#############################################################

# plt_idx= ~pd.isna(df_filt_q_conus["barren_percent"])
# Get Barren percent


# percentage_df_nonveg = get_df_percentage_q(
#     df=df_filt_q_conus,
#     x1_varname="nonveg_percent",
#     x2_varname="AI",
#     y_varname="q_q",
#     weight_by="event_length",
# )
# percentage_df_nonveg.columns
# # %%
# plt.rcParams.update({"font.size": 15})
# fig, ax = plt.subplots(figsize=(7, 4))
# plot_grouped_stacked_bar(
#     ax=ax,
#     df=percentage_df_nonveg,
#     x_column_to_plot="AI_binned2",
#     z_var="nonveg_percent_pct",
#     var_name="rangeland_other",
#     title_name="",
#     weighted=False,
# )
# plt.tight_layout()
# plt.show()

# # %%


# # %%
# from matplotlib.cm import get_cmap


# def sort_percentages(labels):
#     return sorted(labels, key=lambda x: int(x.split("-")[0]))


# def plot_bars(ax, df, x_var, y_var, z_var, title):

#     # Determine unique groups and categories
#     # Define the width of the bars and the space between groups
#     bar_width = 0.2
#     space_between_bars  = 0.025
#     space_between_groups = 0.2

#     # Determine unique values for grouping
#     x_unique = df[x_var].unique()
#     z_unique = sort_percentages(df[z_var].unique())
#     n_groups = len(z_unique)

#     # Generate a green colormap
#     colormap = get_cmap('Greens')
#     color_list = [colormap(i / n_groups) for i in range(n_groups)]

#     # Create the grouped and stacked bars
#     for z_i, z in enumerate(z_unique):
#         for x_i, x in enumerate(x_unique):

#             # Calculate the x position for each group
#             group_offset = (bar_width + space_between_bars) * n_groups
#             x_pos = x_i * (group_offset + space_between_groups) + (bar_width + space_between_bars) * z_i

#             # Get the subset of df for this group
#             subset = df[(df[x_var] == x) & (df[z_var] == z)]

#             # Get bottom values for stacked bars
#             ax.bar(
#                 x_pos,
#                 subset[y_var["column_name"]].median(),
#                 bar_width,
#                 edgecolor='white',
#                 color=color_list[z_i],
#             )

#     # Set the x-ticks to the middle of the groups
#     ax.set_xticks([i * (group_offset + space_between_groups) + group_offset / 2 for i in range(len(x_unique))])
#     ax.set_xticklabels(x_unique, rotation=45)
#     ax.set_xlabel(f"{var_dict["ai_bins"]['label']} {var_dict["ai_bins"]['unit']}")
#     ax.set_ylabel(f"{y_var['label']} {y_var['unit']}, Median")
#     ax.set_title(title)

#     # Set the second x-ticks
#     # Replicate the z_var labels for the number of x_column_to_plot labels
#     z_labels = np.tile(z_unique, len(x_unique))

#     # Adjust the tick positions for the replicated z_var labels
#     new_tick_positions = [i + bar_width / 2 for i in range(len(z_labels))]

#     # Hide the original x-axis ticks and labels
#     ax.tick_params(axis='x', which='both', length=0)

#     # Create a secondary x-axis for the new labels
#     ax2 = ax.twiny()
#     ax2.set_xlim(ax.get_xlim())
#     ax2.set_xticks(new_tick_positions)

#     # Adjust the secondary x-axis to appear below the primary x-axis
#     ax2.spines['top'].set_visible(False)
#     ax2.xaxis.set_ticks_position('bottom')
#     ax2.xaxis.set_label_position('bottom')
#     ax2.spines['bottom'].set_position(('outward', 60))
#     ax2.tick_params(axis='x', which='both', length=0)
#     ax2.set_xticklabels(z_labels, rotation=45)

# plt.rcParams.update({"font.size": 11})

# var_to_test = ["event_length", "q_ETmax", "theta_star", "theta_w", "sm_range", "q_q", "q_delta_theta", "sm_range_abs"]

# for var_name in var_to_test:
#     for i, condition in enumerate(["q>1", "q<1"]):
#         fig, ax = plt.subplots(figsize=(6, 3))
#         if condition == "q<1":
#             subset = df_filt_q_conus[(df_filt_q_conus["q_q"]<1)&~pd.isna(df_filt_q_conus["fractional_wood"])]
#             title = r"$q < 1$"
#         elif condition == "q>1":
#             subset = df_filt_q_conus[(df_filt_q_conus["q_q"]>1)&~pd.isna(df_filt_q_conus["fractional_wood"])]
#             title = r"$q > 1$"
#         bins=[0, 20, 40, 60, 80, 100]
#         labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
#         subset["fractional_wood_pct"] = pd.cut(subset["fractional_wood"], bins=bins, labels=labels)
#         plot_bars(
#             ax=ax,
#             df=subset,
#             y_var=var_dict[var_name],
#             x_var="AI_binned2",
#             z_var="fractional_wood_pct",
#             title=title
#         )

# #%%
# subset.head()

# # %%
# df_filt_q_conus.columns

# # %%
# # # %%
# # longest_events = df_filt_q_conus.sort_values(by='event_length', ascending=False).groupby(['EASE_row_index', 'EASE_column_index']).first().reset_index()
# # group_sizes = df_filt_q_conus.groupby(['EASE_row_index', 'EASE_column_index']).size().reset_index(name='size')
# # filtered_groups = group_sizes[group_sizes['size'] >= 5]

# # # Now, only keep rows from longest_events that have groups with size >= 5
# # longest_events_filtered = longest_events[longest_events.set_index(['EASE_row_index', 'EASE_column_index']).index.isin(filtered_groups.set_index(['EASE_row_index', 'EASE_column_index']).index)]

# # # %%
# # percentage_df_longest_event = get_df_percentage_q(longest_events, "fractional_wood")
# # fig, ax = plt.subplots(figsize=(6, 4))

# # plt.rcParams.update({"font.size": 11})
# # plot_grouped_stacked_bar(
# #     ax=ax,
# #     df=percentage_df_longest_event,
# #     x_column_to_plot="AI_binned2",
# #     z_var="fractional_wood_pct",
# #     var_name="rangeland_wood",
# #     title_name="",
# #     weighted=True
# # )
# # plt.tight_layout()

# # # %%
# # percentage_df_longest_event = get_df_percentage_q(longest_events_filtered, "fractional_wood")
# # fig, ax = plt.subplots(figsize=(6, 4))
# # plot_grouped_stacked_bar(
# #     ax=ax,
# #     df=percentage_df_longest_event,
# #     x_column_to_plot="AI_binned2",
# #     z_var="fractional_wood_pct",
# #     var_name="rangeland_wood",
# #     title_name="",
# #     weighted=False
# # )
# # plt.tight_layout()


# # # %%

# # def plot_q_ai_wood_scatter(df):
# #     fig, ax = plt.subplots(figsize=(6, 4))
# #     sc=ax.scatter(df["fractional_wood"], df["q_q"], c=df["AI"], cmap="RdBu", alpha=0.3)
# #     ax.set_ylim(0,10)
# #     ax.set_xlabel(var_dict["rangeland_wood"]["label"]+" "+var_dict["rangeland_wood"]["unit"])
# #     ax.set_ylabel(var_dict["q_q"]["label"]+" "+var_dict["q_q"]["symbol"])
# #     # Create a colorbar with the scatter plot's color mapping
# #     cbar = plt.colorbar(sc, ax=ax)
# #     cbar.set_label('Aridity Index (MAP/MAE')

# # plot_q_ai_wood_scatter(longest_events_filtered)
# # # %%
# # # Assuming df is the DataFrame with the relevant df and it contains a column named 'df'
# # # for which we want to calculate the coefficient of variation.
# # # Group by 'EASE_row_index' and 'EASE_column_index' and filter groups with count 16

# # # First, group by the indices and filter out the groups with exactly 16 observations
# # long_events = df_filt_q_conus.groupby(['EASE_row_index', 'EASE_column_index']).filter(lambda x: len(x) >=16)

# # # Now, calculate the coefficient of variation for each group
# # def coefficient_of_variation(x):
# #     return x.std() / x.mean()

# # stat = long_events.groupby(['EASE_row_index', 'EASE_column_index']).agg(fractional_wood=("fractional_wood", "median"), AI=("AI", "median"))
# # stat["q_cv"] = long_events.groupby(['EASE_row_index', 'EASE_column_index'])["q_q"].agg(coefficient_of_variation)

# # # %%
# # fig, ax = plt.subplots(figsize=(6, 4))
# # sc=ax.scatter(stat["fractional_wood"], stat["q_cv"], c=stat["AI"], cmap="RdBu", alpha=0.4)
# # ax.set_ylim(0,4)
# # ax.set_xlabel(var_dict["rangeland_wood"]["label"]+" "+var_dict["rangeland_wood"]["unit"])
# # ax.set_ylabel("Coefficient of variation of" +var_dict["q_q"]["symbol"])
# # # Create a colorbar with the scatter plot's color mapping
# # cbar = plt.colorbar(sc, ax=ax)
# # cbar.set_label('Aridity Index (MAP/MAE')
# # # %%
# # from pylab import *
# # def plot_rangeland_map(ax, df, var_item, cmap):
# #     plt.rcParams.update({"font.size": 12})

# #     df = df.drop_duplicates(subset=['latitude', 'longitude'])
# #     pivot_array = df.pivot(
# #         index="latitude", columns="longitude", values=var_item
# #     )
# #     pivot_array[pivot_array.index > -60]

# #     # Get lat and lon
# #     lons = pivot_array.columns.values
# #     lats = pivot_array.index.values

# #     # Plot in the map
# #     custom_cmap = cm.get_cmap(cmap, 5)

# #     im = ax.pcolormesh(
# #         lons, lats, pivot_array, cmap=custom_cmap, transform=ccrs.PlateCarree()
# #     )

# #     ax.set_extent([-160, 170, -60, 90], crs=ccrs.PlateCarree())
# #     ax.coastlines()
# #     ax.set_extent([-125, -66.93457, 24.396308, 49.384358], crs=ccrs.PlateCarree())

# #     # Add colorbar
# #     plt.colorbar(
# #         im,
# #         ax=ax,
# #         orientation="vertical",
# #         shrink=0.35,
# #         pad=0.02,
# #     )

# #     ax.set_xlabel("Longitude")
# #     ax.set_ylabel("Latitude")
# #     ax.set_title(var_item)

# # fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# # plot_rangeland_map(ax, rangeland_info, "fractional_wood", cmap="BuGn")

# # fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
# # plot_rangeland_map(ax, rangeland_info, "fractional_herb", cmap="BuGn")

# # #%%
# # # Assuming rangeland_info is your DataFrame
# # # This will mark all rows that are duplicates as True, keeping the first occurrence as False (not a duplicate by default)
# # duplicates = rangeland_info.duplicated(keep=False)

# # # To show the duplicate rows
# # duplicate_rows = rangeland_info[duplicates]

# # duplicate_rows
# # # %%


# # fig = plt.figure(figsize=(8, 4))

# # plt.rcParams.update({"font.size": 12})
# # ax1 = plt.subplot(121)
# # subset_df = percentage_df[percentage_df["AI_binned2"] == "0-0.5"]
# # plot_fracq_by_pct(
# #     ax1,
# #     subset_df,
# #     "fractional_wood_pct",
# #     "rangeland_wood",
# #     "A.              P/PET < 0.5",
# # )

# # ax1 = plt.subplot(122)
# # subset_df2 = percentage_df[percentage_df["AI_binned2"] == "1.5-"]
# # plot_fracq_by_pct(
# #     ax1,
# #     subset_df2,
# #     "fractional_wood_pct",
# #     "rangeland_wood",
# #     "B.             P/PET > 1.5",
# # )
# # plt.tight_layout()

# # plt.savefig(
# #     os.path.join(fig_dir, f"fracq_fracwood_ai.pdf"), dpi=1200, bbox_inches="tight"
# # )
# # %%

# # %% Loss function plots + parameter scatter plots

# # # %%  ##########################
# # ## Loss function plots + parameter scatter plots  (Figure 4)
# # ##########################
# # save = True
# # # Create a figure with subplots
# # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# # # mpl.rcParams['font.size'] = 18
# # # # plt.rcParams.update({'axes.labelsize' : 14})
# # # plt.rcParams.update({"font.size": 18})
# # # Plotting
# # plot_loss_func_categorical(
# #     axs[0],
# #     df_filt_q,
# #     var_dict["veg_class"],
# #     categories=vegetation_color_dict.keys(),
# #     colors=list(vegetation_color_dict.values()),
# #     title=None,
# #     plot_legend=False,
# # )

# # plot_scatter_with_errorbar_categorical(
# #     ax=axs[1],
# #     df=df_filt_q,
# #     x_var=var_dict["ai"],
# #     y_var=var_dict["q_q"],
# #     z_var=var_dict["veg_class"],
# #     categories=list(vegetation_color_dict.keys()),
# #     colors=list(vegetation_color_dict.values()),
# #     quantile=25,
# #     title=None,
# #     plot_logscale=False,
# #     plot_legend=False,
# # )

# # axs[1].set_ylim([0, 0.75])
# # axs[1].set_xlim([0, 1.0])

# # # axs[0].set_yticks(
# #     [-0.1, -0.08, -0.06, -0.04, -0.02, 0.0], ["0.10", 0.08, 0.06, 0.04, 0.02, "0.00"]
# # )
# # axs[1].set_yticks(
# #     [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], ["", 1.0, "", 2.0, "", 3.0, "", 4.0]
# # )


# plt.tight_layout()
# plt.show()

# if save:
#     # Save the combined figure
#     fig.savefig(
#         os.path.join(fig_dir, "q_veg_ai.png"),
#         dpi=1200,
#         bbox_inches="tight",
#         transparent=True,
#     )
#     fig.savefig(
#         os.path.join(fig_dir, "q_veg_ai.pdf"),
#         dpi=1200,
#         bbox_inches="tight",
#         transparent=True,
#     )
#     fig.savefig(
#         os.path.join(fig_dir, "q_veg_ai.svg"),
#         dpi=1200,
#         bbox_inches="tight",
#         transparent=True,
#     )
# %%
# # %%
# # With variabiliity
# plt.rcParams.update({"font.size": 18})
# fig, axs = plt.subplots(2, 2, figsize=(8, 8))
# plot_loss_func(
#     axs[0, 0],
#     df_filt_q_agg,
#     var_dict["veg_class_mode"],
#     categories=vegetation_color_dict.keys(),
#     colors=list(vegetation_color_dict.values()),
#     plot_legend=False,
#     title="(a)",
#     median_by_pixel=True
# )

# plot_scatter_with_errorbar(
#     ax=axs[0, 1],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_theta_star_var"],
#     y_var=var_dict["q_q_var"],
#     z_var=var_dict["veg_class_mode"],
#     quantile=25,
#     categories=list(vegetation_color_dict.keys()),
#     colors=list(vegetation_color_dict.values()),
#     title="(b)",
# )

# plot_scatter_with_errorbar(
#     ax=axs[1, 0],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_ETmax_var"],
#     y_var=var_dict["q_q_var"],
#     z_var=var_dict["veg_class_mode"],
#     quantile=25,
#     categories=list(vegetation_color_dict.keys()),
#     colors=list(vegetation_color_dict.values()),
#     title="(c)",
# )
# plot_scatter_with_errorbar(
#     ax=axs[1, 1],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_theta_star_var"],
#     y_var=var_dict["q_ETmax_var"],
#     z_var=var_dict["veg_class_mode"],
#     quantile=25,
#     categories=list(vegetation_color_dict.keys()),
#     colors=list(vegetation_color_dict.values()),
#     title="(d)",
# )

# plt.tight_layout()
# plt.show()

# if save:
#     # Save the combined figure
#     fig.savefig(
#         os.path.join(fig_dir, "sup_lossfnc_veg_var_by_pixel.png"), dpi=1200, bbox_inches="tight"
#     )
#     fig.savefig(
#         os.path.join(fig_dir, "sup_lossfnc_veg_var_by_pixel.pdf"), dpi=1200, bbox_inches="tight"
#     )
# # fig.savefig(os.path.join(fig_dir, "sup_lossfnc_veg_legend.pdf"), dpi=1200, bbox_inches="tight")


# # %%
# # Aridity Index - median
# fig, axs = plt.subplots(2, 2, figsize=(8, 8))
# plt.rcParams.update({"font.size": 18})

# plot_loss_func(
#     ax=axs[0, 0],
#     df=df_filt_q_agg,
#     z_var=var_dict["ai_bins"],
#     cmap=ai_cmap,
#     plot_legend=False,
#     title="(a)",
#     median_by_pixel=True,
# )

# plot_scatter_with_errorbar(
#     ax=axs[0, 1],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_theta_star_median"],
#     y_var=var_dict["q_q_median"],
#     z_var=var_dict["ai_bins"],
#     cmap=ai_cmap,
#     quantile=25,
#     title="(b)",
#     plot_logscale=False,
#     plot_legend=False,
# )

# plot_scatter_with_errorbar(
#     ax=axs[1, 0],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_ETmax_median"],
#     y_var=var_dict["q_q_median"],
#     z_var=var_dict["ai_bins"],
#     cmap=ai_cmap,
#     quantile=25,
#     title="(c)",
#     plot_logscale=False,
#     plot_legend=False,
# )
# plot_scatter_with_errorbar(
#     ax=axs[1, 1],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_ETmax_median"],
#     y_var=var_dict["q_theta_star_median"],
#     z_var=var_dict["ai_bins"],
#     cmap=ai_cmap,
#     quantile=25,
#     title="(d)",
#     plot_logscale=False,
#     plot_legend=False,
# )

# plt.tight_layout()
# plt.show()

# # Save the combined figure

# if save:
#     save_figure(fig, fig_dir, f"sup_lossfnc_ai_median_by_pixel", "png", 1200)
#     save_figure(fig, fig_dir, f"sup_lossfnc_ai_median_by_pixel", "pdf", 1200)


# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_ai_legend.png"), dpi=1200, bbox_inches="tight")
# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_ai_legend.pdf"), dpi=1200, bbox_inches="tight")


# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_sand_legend.png"), dpi=1200, bbox_inches="tight")
# fig.savefig(os.path.join(fig_dir, "sup_lossfnc_sand_legend.pdf"), dpi=1200, bbox_inches="tight")
# # %%
# # sand - median

# fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# plot_loss_func(
#     ax=axs[0, 0],
#     df=df_filt_q_agg,
#     z_var=var_dict["sand_bins"],
#     cmap=sand_cmap,
#     plot_legend=False,
#     title="(a)",
#     median_by_pixel=True,
# )

# plot_scatter_with_errorbar(
#     ax=axs[0, 1],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_theta_star_median"],
#     y_var=var_dict["q_q_median"],
#     z_var=var_dict["sand_bins"],
#     cmap=sand_cmap,
#     quantile=25,
#     title="(b)",
#     plot_logscale=False,
#     plot_legend=False,
# )

# plot_scatter_with_errorbar(
#     ax=axs[1, 0],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_ETmax_median"],
#     y_var=var_dict["q_q_median"],
#     z_var=var_dict["sand_bins"],
#     cmap=sand_cmap,
#     quantile=25,
#     title="(c)",
#     plot_logscale=False,
#     plot_legend=False,
# )
# plot_scatter_with_errorbar(
#     ax=axs[1, 1],
#     df=df_filt_q_agg,
#     x_var=var_dict["q_ETmax_median"],
#     y_var=var_dict["q_theta_star_median"],
#     z_var=var_dict["sand_bins"],
#     cmap=sand_cmap,
#     quantile=25,
#     title="(d)",
#     plot_logscale=False,
#     plot_legend=False,
# )

# plt.tight_layout()
# plt.show()

# if save:
#     save_figure(fig, fig_dir, f"sup_lossfnc_sand_median", "png", 1200)
#     save_figure(fig, fig_dir, f"sup_lossfnc_sand_median", "pdf", 1200)
