from pathlib import Path
from typing import Callable, Any, Union

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.stats import norm

from reardon import RACES, run, RACE_WEIGHT, RES_LOC, RES_STD


def run_agg_colls(num_repeats=10, **kwargs):
    """
    College dataframe as an average of multiple runs
    """
    runs_ = Parallel(n_jobs=-1, verbose=1)(
        delayed(run)(**kwargs) for _ in range(num_repeats)
    )
    colls, _, _, _ = zip(*runs_)
    # average of runs
    colls = np.rec.fromarrays(
        [
            np.mean([colls_[name] for colls_ in colls], axis=0)
            for name in colls[0].dtype.names
        ],
        dtype=colls[0].dtype,
    )
    # as dataframe
    ind_yc = pd.MultiIndex.from_product(
        (range(colls.shape[0]), range(colls.shape[1])), names=["year", "coll"]
    )
    colls = pd.DataFrame(colls.ravel(), index=ind_yc)
    return colls


def pct_enroll_by_race(**kwargs):
    """
    Percentage of enrolled students by race for active colleges after year 25
    """
    _, _, _, enr = run(**kwargs)
    enr = enr[(enr.year >= 25) & enr.active]
    races, counts = np.unique(enr.race, return_counts=True)
    pct = counts * 100 / np.sum(counts)
    names = np.take(RACES, races)
    return pd.Series(pct, index=names)


# mixture of resources distribution
res_loc_ = np.sum(RACE_WEIGHT * RES_LOC)
res_var = np.sum(RACE_WEIGHT * (RES_STD**2 + RES_LOC**2 - res_loc_**2))
# common resource quintiles for all runs
res_bins = norm.ppf(np.linspace(0, 1, 5 + 1), loc=res_loc_, scale=np.sqrt(res_var))


def pct_enroll_by_res(**kwargs):
    """
    Percentage of enrolled students by resource quintile for active colleges after year 25
    """
    _, _, _, enr = run(**kwargs)
    enr = enr[(enr.year >= 25) & enr.active]
    counts, _ = np.histogram(enr.res, bins=res_bins)
    pct = counts * 100 / np.sum(counts)
    q = [f"Q{i}" for i in range(1, 5 + 1)]
    return pd.Series(pct, index=q)


def df_param_grid(param_grid):
    return pd.MultiIndex.from_product(
        param_grid.values(), names=param_grid.keys()
    ).to_frame(index=False)


def df_exp_grid(param_grid, exp):
    df = df_param_grid(param_grid)
    runs = Parallel(n_jobs=-1, verbose=1)(
        delayed(exp)(**conf) for _, conf in df.iterrows()
    )
    df = df.join(pd.concat(runs, axis=1).T)
    return df


def df_melt_compo_by(df, var_name):
    # id_vars = df.filter(like="wgt_").columns
    # df = df.melt(
    #     id_vars=id_vars,
    #     var_name=var_name,
    #     value_name="Composition (%)",
    # )
    df = df.rename(
        columns={
            "wgt_aa_race": "Race weight",
            "wgt_aa_ses": "SES weight",
            "wgt_recr_race": "Recruit weight",
        }
    )
    return df


def df_compo_by_race(param_grid):
    """
    Racial composition of active colleges after year 25
    """
    df = df_exp_grid(param_grid, pct_enroll_by_race)
    df = df_melt_compo_by(df, "Race")
    return df


def df_compo_by_res(param_grid):
    """
    SES composition of active colleges after year 25
    """
    df = df_exp_grid(param_grid, pct_enroll_by_res)
    df = df_melt_compo_by(df, "Resources")
    return df


def plot_pct_minority(coll):
    f, ax = plt.subplots()
    sns.lineplot(
        x="year",
        y="pct_minority",
        hue="active",
        units="coll",
        data=coll,
        estimator=None,
        legend=False,
        lw=0.5,
        ax=ax,
    )
    sns.lineplot(
        x="year",
        y="pct_minority",
        hue="active",
        data=coll,
        legend=False,
        lw=5,
        ci=None,
        ax=ax,
    )
    ax.set(xlabel="Year", ylabel="Percent minority")
    return f


def plot_quality(coll):
    f, ax = plt.subplots()
    sns.lineplot(
        x="year",
        y="qual",
        hue="active",
        units="coll",
        data=coll,
        estimator=None,
        legend=False,
        lw=0.5,
        ax=ax,
    )
    sns.lineplot(
        x="year", y="qual", hue="active", data=coll, legend=False, lw=5, ci=None, ax=ax
    )
    ax.set(xlabel="Year", ylabel="College quality")
    return f


def my_quiver(ax, rows, x, y, z, t0, t1):
    for _, row in rows:
        ax.annotate(
            "",
            xy=(row[(x, t1)], row[(y, t1)]),
            xytext=(row[(x, t0)], row[(y, t0)]),
            arrowprops=dict(arrowstyle="->", ec=sns.color_palette()[int(row[(z, t0)])]),
        )


def plot_minority_mean_ach(coll):
    df = coll[coll.index.get_level_values("year").isin([14, 29])].reset_index()
    f, ax = plt.subplots()
    sns.scatterplot(
        x="ach_mean", y="pct_minority", hue="active", data=df, legend=False, s=0, ax=ax
    )
    my_quiver(
        ax,
        df.set_index(["year", "coll"]).unstack("year").iterrows(),
        "ach_mean",
        "pct_minority",
        "active",
        14,
        29,
    )
    ax.set(
        xlabel="Mean achievement of enrolled students",
        ylabel="Proportion minority students",
    )
    return f


def make_df_fig_2():
    df_fig2 = df_compo_by_race(
        {"wgt_aa_ses": [0, 50, 100, 150], "wgt_recr_race": [0, 25, 50, 100]}
    )

    # as a share of real-world AA
    df_fig2.rename(columns={"Composition (%)": "Rel. enrollment"}, inplace=True)
    real_world = pct_enroll_by_race(wgt_aa_race=260)
    df_fig2[real_world.index] = df_fig2[real_world.index] / real_world

    # only Black and Hispanic
    df_fig2 = df_fig2.drop(columns=["Asian", "White"])

    return df_fig2


def create_if_not_exists(
    p: Union[str, Path], func: Callable[[Any], pd.DataFrame], **kwargs
):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file():
        x = func(**kwargs)
        x.to_pickle(str(p))
    else:
        x = pd.read_pickle(p)
    return x


def make_all():
    path = Path("data")
    path.mkdir(parents=True, exist_ok=True)

    coll_0_0_0 = create_if_not_exists(
        path / "coll_0_0_0.pkl", run_agg_colls, num_years=50
    )

    coll_260_0_0 = create_if_not_exists(
        path / "coll_260_0_0.pkl", run_agg_colls, num_years=50, wgt_aa_race=260
    )

    coll_0_150_100 = create_if_not_exists(
        path / "coll_0_150_100.pkl",
        run_agg_colls,
        num_years=50,
        wgt_aa_ses=150,
        wgt_recr_race=100,
    )

    coll_0_150_100_f1 = create_if_not_exists(
        path / "coll_0_150_100_f1.pkl",
        run_agg_colls,
        num_years=50,
        wgt_aa_ses=150,
        wgt_recr_race=100,
        aa_formula1=True,
    )

    coll_0_150_100_pen = create_if_not_exists(
        path / "coll_0_150_100_pen.pkl",
        run_agg_colls,
        num_years=50,
        wgt_aa_ses=150,
        wgt_recr_race=100,
        aa_penalize=True,
    )

    coll_0_150_100_pq07 = create_if_not_exists(
        path / "coll_0_150_100_pq07.pkl",
        run_agg_colls,
        num_years=50,
        wgt_aa_ses=150,
        wgt_recr_race=100,
        percept_qual_07=True,
    )

    df_fig_2 = create_if_not_exists(path / "fig2.pkl", make_df_fig_2)

    df_fig_a2 = create_if_not_exists(
        path / "figA2.pkl",
        df_compo_by_race,
        param_grid={"wgt_aa_ses": [0, 75, 150], "wgt_recr_race": [25, 50, 100]},
    )

    df_fig_a3 = create_if_not_exists(
        path / "figA3.pkl",
        df_compo_by_res,
        param_grid={"wgt_aa_ses": [0, 75, 150], "wgt_recr_race": [25, 50, 100]},
    )

    df_fig_a4 = create_if_not_exists(
        path / "figA4.pkl",
        df_compo_by_race,
        param_grid={"wgt_aa_ses": [0, 75, 150], "wgt_aa_race": [0, 150, 300]},
    )

    df_fig_a5 = create_if_not_exists(
        path / "figA5.pkl",
        df_compo_by_res,
        param_grid={"wgt_aa_ses": [0, 75, 150], "wgt_aa_race": [0, 150, 300]},
    )

    return (
        coll_0_0_0,
        coll_260_0_0,
        coll_0_150_100,
        coll_0_150_100_f1,
        coll_0_150_100_pen,
        coll_0_150_100_pq07,
        df_fig_2,
        df_fig_a2,
        df_fig_a3,
        df_fig_a4,
        df_fig_a5,
    )
