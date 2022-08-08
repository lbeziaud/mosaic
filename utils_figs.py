from itertools import product, starmap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory
from scipy.stats import norm

from model import RACE_WEIGHTS, RES_LOCS, RES_SCALES
from model import run as run_uncached

sns.set_style("whitegrid")

# setup cached model run
location = "./cache"
memory = Memory(location, verbose=0, mmap_mode="r")
run = memory.cache(run_uncached)

# number of time each run is repeated
n_repeats = 10

# initial seed (use to generate subsequent seeds)
entropy = 286130738418515439860296005820123389689

# number of quantile to consider for students' resources
n_res_bins = 5
# mixture of resources distribution
res_loc_mix = np.sum(RACE_WEIGHTS * RES_LOCS)
res_var_mix = np.sum(
    RACE_WEIGHTS * (RES_SCALES**2 + RES_LOCS**2 - res_loc_mix**2)
)
# prepare common resources quintiles for all runs
res_bins = norm.ppf(
    np.linspace(0, 1, n_res_bins + 1), loc=res_loc_mix, scale=np.sqrt(res_var_mix)
)

# order used to plot races
order_race = ["Black", "Hispanic", "Asian", "White"]
# order used to plot resources quintiles
order_res = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def dict_product(d):
    """Cartesian product of input iterables with keys."""
    # dict_product({"A": [1, 2], "B": [3]}) --> [{'A': 1, 'B': 3}, {'A': 2, 'B': 3}]
    return [dict(zip(d.keys(), v)) for v in product(*d.values())]


def run_repeat(**params):
    """Run (cached) model multiple times with distinct seeds. Initial seed is constant."""
    sq = np.random.SeedSequence(entropy)
    seeds = sq.generate_state(n_repeats)
    return (run(**params, seed=seed) for seed in seeds)


def run_repeat_colleges(**params):
    """Colleges average by year and college of multiple runs"""
    results = run_repeat(**params)
    colleges, _, _ = zip(*results)
    colleges = pd.concat(colleges).groupby(["year", "coll"]).mean().reset_index()
    return colleges


def run_repeat_agg(func, **params):
    """Average aggreation of func mapped on multiple runs."""
    return pd.concat(starmap(func, run_repeat(**params))).groupby(level=0).mean()


def grid_run(param_grid, func):
    """Multiple aggregated model runs for parameter combinations with parameters as index."""
    paramspace = dict_product(param_grid)

    dfs = (run_repeat_agg(func, **params) for params in paramspace)

    return pd.concat(
        dfs,
        keys=(tuple(params.values()) for params in paramspace),
        names=param_grid.keys(),
    )


def pct_race_enroll(colleges, students, outcomes):
    """Percentage of total enrolled students after year 25 in active colleges by race."""
    return (
        outcomes[outcomes.enrollment]
        .xs(slice(25, None), level="year", drop_level=False)
        .join(colleges[colleges.active], how="inner")
        .join(students)
        .race.value_counts(normalize=True, sort=False)
        .mul(100)
        .rename_axis("race")
        .rename("pct_enroll")
    )


def pct_ses_enroll(colleges, students, outcomes):
    """Percentage of total enrolled students after year 25 in active colleges by resources quintile."""
    return (
        pd.cut(
            outcomes[outcomes.enrollment]
            .xs(slice(25, None), level="year", drop_level=False)
            .join(colleges[colleges.active], how="inner")
            .join(students)
            .res,
            bins=res_bins,
            labels=[f"Q{i}" for i in range(1, n_res_bins + 1)],
        )
        .value_counts(normalize=True, sort=False)
        .mul(100)
        .rename_axis("res")
        .rename("pct_enroll")
    )


def changes_bh_enrollment(**params):
    """Plot change in Black and Hispanic enrollments."""
    colleges = run_repeat_colleges(**params)
    fig, ax = plt.subplots()
    sns.lineplot(
        x="year",
        y="pct_minority",
        hue="active",
        units="coll",
        data=colleges,
        estimator=None,
        legend=False,
        lw=0.5,
        palette=["lightgray", "black"],
        ax=ax,
    )
    sns.lineplot(
        x="year",
        y="pct_minority",
        hue="active",
        data=colleges,
        legend=False,
        lw=3,
        ci=None,
        ax=ax,
        palette=["darkgray", "black"],
    )
    ax.set(xlabel="Year", ylabel="Percent minority")
    return fig


def changes_coll_qual(**params):
    """Plot change in colleges's quality."""
    colleges = run_repeat_colleges(**params)
    fig, ax = plt.subplots()
    sns.lineplot(
        x="year",
        y="qual",
        hue="active",
        units="coll",
        data=colleges,
        estimator=None,
        legend=False,
        lw=0.5,
        palette=["lightgray", "black"],
        ax=ax,
    )
    sns.lineplot(
        x="year",
        y="qual",
        hue="active",
        data=colleges,
        legend=False,
        lw=3,
        ci=None,
        ax=ax,
        palette=["darkgray", "black"],
    )
    ax.set(xlabel="Year", ylabel="College quality")
    return fig


def my_quiver(ax, df, x, y, z, t0, t1):
    """Plot arrows from xy at t0 to xy at t1 with color z."""
    rows = df.set_index(["year", "coll"]).unstack("year").iterrows()
    for coll, row in rows:
        ax.annotate(
            "",
            xy=(row[(x, t1)], row[(y, t1)]),
            xytext=(row[(x, t0)], row[(y, t0)]),
            arrowprops=dict(
                arrowstyle="->",
                ec=["darkgray", "black"][int(row[(z, t0)])] if coll != -1 else "red",
            ),
        )


def coll_shift_aux(colleges, students, outcomes):
    """Aggregated view of colleges at years 14 and 29."""
    outcomes = outcomes[outcomes.index.get_level_values("year").isin([14, 29])]

    wo_enroll = (
        students.loc[outcomes[~outcomes.enrollment][[]].loc[:, 0, :].index]
        .assign(coll=-1)
        .set_index("coll", append=True)
        .assign(active=False)
    )

    return (
        pd.concat(
            [
                outcomes[outcomes.enrollment][[]]
                .join(colleges[["active"]])
                .join(students),
                wo_enroll,
            ]
        )
        .groupby(["year", "coll"])
        .agg(
            {
                "active": "first",
                "ach": "mean",
                "race": lambda s: s.isin(["Black", "Hispanic"]).mean(),
                "res": lambda s: s.le(res_bins[2 + 1]).mean(),  # bottom two quintiles
            }
        )
        .reset_index()
    )


def plot_d123(scenario, y):
    """Plot figures D1, D2, D3."""
    nrows = len(scenario) // 2

    fig, axes = plt.subplots(ncols=2, nrows=nrows, sharex=True, sharey=True)

    for ax, (title, df) in zip(axes.flat, scenario.items()):
        ax.set_title(title, fontsize=8)
        ax.set(xlim=(750, 1450), ylim=(0, 0.69))
        my_quiver(ax, df, "ach", y, "active", 14, 29)

    fig.tight_layout()

    return fig
