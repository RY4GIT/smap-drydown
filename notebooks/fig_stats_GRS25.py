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
data_dir = r"G:\Shared drives\Ryoko and Hilary\smap-drydown\data"

# Read the model output (results)
output_dir = r"G:\Shared drives\Ryoko and Hilary\smap-drydown\output"
results_file = rf"all_results_processed.csv"

datarods_dir = "datarods"
anc_rangeland_processed_file = "anc_info_rangeland_processed.csv"
coord_info_file = "coord_info.csv"

################ CHANGE HERE FOR PLOT VISUAL CONFIG & SETTING VARIABLES #########################

## Define parameters
save = True

note_dir = r"C:\Users\flipl\dev\smap-drydown\notebooks"
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


count_model_performance(df_filt_q_or_exp, "diff_aic_q_tauexp")
count_model_performance(df_filt_q_or_exp, "diff_aic_q_exp")

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

    # Remove all spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_frame_on(False)  # Remove the frame completely

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
    ax.coastlines(linewidth=0.5, color="grey")

    if not bar_label:
        bar_label = f'{var_item["symbol"]} {var_item["unit"]}'

    # Add colorbar
    plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        # label=f'{stat_label} {var_item["label"]} {var_item["unit"]}',
        label=bar_label,
        shrink=0.1,
        # width=0.1,
        pad=0.01,
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
var_key_exp = "diff_aic_q_exp"

norm_exp = Normalize(
    vmin=var_dict[var_key_exp]["lim"][0], vmax=var_dict[var_key_exp]["lim"][1]
)

# Plot and save maps for exp model
plt.rcParams.update({"font.size": 12})
fig_map_aic, ax = plt.subplots(
    figsize=(9, 9), subplot_kw={"projection": ccrs.Robinson()}
)
plot_map(
    ax=ax,
    df=df_filt_q_and_exp,
    coord_info=coord_info,
    cmap="RdBu_r",
    norm=norm_exp,
    var_item=var_dict[var_key_exp],
    stat_type=stat_type,
    bar_label="$AIC$",
)
if save:
    save_figure(fig_map_aic, fig_dir, f"aic_map_{stat_type}_and_exp", "png", 900)

# Print statistical summaries for exp model
print_global_stats(df_filt_q_and_exp, "diff_aic_q_exp", "nonlinear - linear")

# %%
# Setup common variables
var_key_exp = "diff_R2_exp"

norm_exp = Normalize(
    vmin=var_dict[var_key_exp]["lim"][0], vmax=var_dict[var_key_exp]["lim"][1]
)

# Plot and save maps for exp model
plt.rcParams.update({"font.size": 12})
fig_map_aic, ax = plt.subplots(
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
    bar_label="$R^2$",
)
if save:
    save_figure(fig_map_aic, fig_dir, f"R2_map_{stat_type}_and_exp", "png", 900)

# Print statistical summaries for exp model
print_global_stats(df_filt_q_and_exp, "diff_R2_q_exp", "nonlinear - linear")

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
    plot_norm_theta=False,
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

        norm_theta = (theta - theta_w) / (theta_star - theta_w)
        norm_dtheta = dtheta / ETmax

        if plot_norm_theta:
            plot_theta = norm_theta
        else:
            plot_theta = theta

        if plot_norm_theta:
            plot_dtheta = norm_dtheta
        else:
            plot_dtheta = dtheta

        # Plot loss function from median parameters
        if plot_norm_theta:
            linewidth = 1.0
        else:
            linewidth = 1.5
        ax.plot(
            plot_theta,
            plot_dtheta,
            label=f"{category}",
            color=colors[i],
            linewidth=linewidth,
        )

    ax.invert_yaxis()

    if plot_norm_theta:
        ax.set_xlabel(
            r"$\frac{\theta - \theta_w}{\theta^* - \theta_w}$"
        )
        ax.set_ylabel(r"$(d\theta/dt)/ET_{\mathrm{max}}$")
    else:
        ax.set_xlabel(
            f"{var_dict['theta']['label']}\n{var_dict['theta']['symbol']} {var_dict['theta']['unit']}"
        )
        ax.set_ylabel(r"$d\theta/dt$" + f" {var_dict['dtheta']['unit']}")
    # if title == "":
    #     title = f'Median loss function by {z_var["label"]} {z_var["unit"]}'
    # ax.set_title(title, loc="left", fontsize=14)

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



# Aridity
plt.rcParams.update({"font.size": 12})
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plot_loss_func(
    axs[1],
    df=df_filt_q_agg,
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    plot_legend=False,
    median_by_pixel=True,
    plot_norm_theta=True,
)
plot_loss_func(
    axs[0],
    df=df_filt_q_agg,
    z_var=var_dict["ai_bins"],
    cmap=ai_cmap,
    plot_legend=False,
    median_by_pixel=True,
    plot_norm_theta=False,
)


plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"fig4_lossfnc_ai", "png", 300)
    save_figure(fig, fig_dir, f"fig4_lossfnc_ai", "pdf", 300)

# %%
# Sand
plt.rcParams.update({"font.size": 12})
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plot_loss_func(
    ax=axs[0],
    df=df_filt_q_agg,
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    plot_legend=False,
    # title="(b)",
    median_by_pixel=True,
    plot_norm_theta=False,
)
plot_loss_func(
    ax=axs[1],
    df=df_filt_q_agg,
    z_var=var_dict["sand_bins"],
    cmap=sand_cmap,
    plot_legend=False,
    # title="(b)",
    median_by_pixel=True,
    plot_norm_theta=True,
)


plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"fig4_lossfnc_sand", "png", 1200)
    save_figure(fig, fig_dir, f"fig4_lossfnc_sand", "pdf", 1200)

# %%
# Vegetation
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
plot_loss_func(
    axs[0],
    df_filt_q_agg,
    var_dict["veg_class_mode"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    plot_legend=False,
    # title="(c)",
    median_by_pixel=True,
    plot_norm_theta=False,
)
plot_loss_func(
    axs[1],
    df_filt_q_agg,
    var_dict["veg_class_mode"],
    categories=vegetation_color_dict.keys(),
    colors=list(vegetation_color_dict.values()),
    plot_legend=False,
    # title="(c)",
    median_by_pixel=True,
    plot_norm_theta=True,
)
plt.tight_layout()
plt.show()

if save:
    save_figure(fig, fig_dir, f"fig4_lossfnc_veg", "png", 1200)
    save_figure(fig, fig_dir, f"fig4_lossfnc_veg", "pdf", 1200)

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
        ax.set_ylabel("Weighted proportion of\ndrydown events by duration ()")
        ax.set_ylim([0, 15])
    else:
        ax.set_ylabel("% drydown events")
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


plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(7, 6))
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
save_figure(fig, fig_dir, "fracwood_q_unweighted", "png", 1200)





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
