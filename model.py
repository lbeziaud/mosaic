from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression

N_YEARS = 30

N_COLLS = 40
N_STUDS = 10000

N_ACTIVE = 4

N_SEATS = 150

QUAL_LOC, QUAL_SCALE = 1070, 130

WGT_AA_RACE = 0
WGT_AA_SES = 0
WGT_RECRUIT = 0

RACE_NAMES = ["Asian", "Black", "Hispanic", "White"]

RACE_WEIGHTS = np.array([0.05, 0.15, 0.2, 0.6])

RES_LOCS = np.array([0.012, -0.224, -0.447, 0.198])
RES_SCALES = np.array([0.833, 0.666, 0.691, 0.657])

ACH_LOCS = np.array([1038, 869, 895, 1052])
ACH_SCALES = np.array([202, 169, 185, 186])

AR_CORRS = np.array([0.441, 0.305, 0.373, 0.395])


S = namedtuple("S", ("colleges", "students", "outcomes"))


def run(
    *,
    n_years=N_YEARS,
    n_colls=N_COLLS,
    n_studs=N_STUDS,
    n_active=N_ACTIVE,
    n_seats=N_SEATS,
    qual_loc=QUAL_LOC,
    qual_scale=QUAL_SCALE,
    wgt_aa_race=WGT_AA_RACE,
    wgt_aa_ses=WGT_AA_SES,
    wgt_recruit=WGT_RECRUIT,
    race_weights=RACE_WEIGHTS,
    res_locs=RES_LOCS,
    res_scales=RES_SCALES,
    ach_locs=ACH_LOCS,
    ach_scales=ACH_SCALES,
    ar_corrs=AR_CORRS,
    seed=None,
    ses_alt=False,
    ses_penalty=False,
    obs_qual_alt=False,
):
    """

    :param n_years: number of years to simulate
    :param n_colls: number of colleges
    :param n_studs: number of students
    :param n_active: number of colleges implementing policies
    :param n_seats: number of seats per colleges
    :param qual_loc: mean of colleges' quality normal distribution
    :param qual_scale: scale of colleges' quality normal distribution
    :param wgt_aa_race: weight of race-based afformative action
    :param wgt_aa_ses: weight of SES-based affirmative action
    :param wgt_recruit: weight of race-based recruiting
    :param race_weights: weights of racial distribution
    :param res_locs: means of students' resources normal distribution
    :param res_scales: scales of students' resources normal distribution
    :param ach_locs: means of students' achievement normal distribution
    :param ach_scales: scales of students' achievement normal distribution
    :param ar_corrs: correlations between students' resources and achievement
    :param seed:
    :param ses_alt: alternative SES-based AA
    :param ses_penalty: alternative SES-based AA
    :param obs_qual_alt: alternative perception quality upper-bound
    :return: namedtuple of dataframes with results for colleges, students and decisions
    """
    # weighted mixture
    ach_loc_mix = np.sum(race_weights * ach_locs)
    ach_var_mix = np.sum(
        race_weights * (ach_scales**2 + ach_locs**2 - ach_loc_mix**2)
    )

    rng = np.random.default_rng(seed)

    qual = np.empty((n_years, n_colls))
    qual[0] = rng.normal(qual_loc, qual_scale, size=n_colls)
    qual[0, ::-1].sort()  # sort by descending quality

    # mask true if college (top n_active) will use policy
    active = np.zeros(n_colls, dtype=bool)
    active[:n_active] = True

    expected_yield = 0.2 + 0.6 * np.argsort(qual[0])[::-1] / (n_colls - 1)

    race = rng.choice(4, size=(n_years, n_studs), p=race_weights)
    minority = (race == 1) | (race == 2)  # Black or Hispanic

    res = rng.normal(res_locs[race], res_scales[race])

    # generate ach such that N(ach_loc, ach_std^2) and corr(res, ach) = achres_corr
    b = ar_corrs * ach_scales / res_scales
    a = ach_locs - b * res_locs
    esd = np.sqrt(ach_scales**2 - b**2 * res_scales**2)
    ach = a[race] + b[race] * res + esd[race] * rng.normal(size=(n_years, n_studs))

    bonus_ach = 0.1 * res * ach_scales[race]

    stud_obs_ach_rel = np.clip(0.7 + 0.1 * res, 0.5, 0.9)
    stud_obs_ach_err_std = np.sqrt(
        ach_var_mix * (1 - stud_obs_ach_rel) / stud_obs_ach_rel
    )
    stud_obs_ach_err = rng.normal(0, stud_obs_ach_err_std, (n_years, n_studs))
    stud_obs_ach = ach + bonus_ach + stud_obs_ach_err

    coll_obs_ach_err_std = np.sqrt(ach_var_mix * (1 - 0.8) / 0.8)
    coll_obs_ach_err = rng.normal(0, coll_obs_ach_err_std, (n_years, n_colls, n_studs))
    coll_obs_ach = (ach + bonus_ach)[:, np.newaxis, :] + coll_obs_ach_err

    stud_obs_qual_err_ = rng.normal(size=(n_years, n_colls, n_studs))  # pre-draw noise

    application = np.zeros((n_years, n_colls, n_studs), dtype=bool)
    admission = np.zeros_like(application)
    enrollment = np.zeros_like(application)

    admit_lr = LogisticRegression(
        warm_start=True, random_state=np.random.RandomState(seed)
    )
    admit_lr.coef_ = np.array([[0.015]])
    admit_lr.intercept_ = np.array([0.0])
    admit_lr.classes_ = np.array([0, 1])

    for year in range(n_years):
        # update quality
        if year > 0:
            # mean achievement of last enrolled students
            num_enroll = enrollment[year - 1].sum(axis=1)
            mean_coll_ach = enrollment[year - 1] @ ach[year - 1]
            mean_coll_ach = np.divide(
                mean_coll_ach, num_enroll, out=mean_coll_ach, where=num_enroll != 0
            )
            qual[year] = 0.9 * qual[year - 1] + 0.1 * mean_coll_ach

        # update expected_yield
        if year > 0:
            # up to 3 years of adm / enroll
            last3 = slice(max(0, year - 3), year)
            num_enroll_3 = enrollment[last3].sum(axis=(0, 2))
            num_admit_3 = admission[last3].sum(axis=(0, 2))
            # 0.0001 if 0 enroll
            expected_yield = np.divide(
                num_enroll_3,
                num_admit_3,
                out=np.full(n_colls, 0.0001),
                where=num_enroll_3 != 0,
            )

        # update stud_obs_qual
        stud_qual_rel_max = 0.7 if obs_qual_alt else 0.9
        stud_obs_qual_rel = np.clip(0.7 + 0.1 * res[year], 0.5, stud_qual_rel_max)
        stud_obs_qual_err_std = np.sqrt(
            np.var(qual[year])
            * ((1 - stud_obs_qual_rel) / stud_obs_qual_rel)[np.newaxis, :]
        )
        stud_obs_qual_err = stud_obs_qual_err_std * stud_obs_qual_err_[year]
        stud_obs_qual = qual[year, :, np.newaxis] + stud_obs_qual_err

        # update admit_reg
        if year > 5:
            last5 = slice(year - 5, year)
            mask = application[last5]
            y_train = admission[last5][mask]
            x_train = ach[last5, np.newaxis, :] - qual[last5, :, np.newaxis]
            x_train = x_train[mask].reshape(-1, 1)
            admit_lr.fit(x_train, y_train)

        util = -250 + stud_obs_qual

        admit_prob = admit_lr.predict_proba(
            (stud_obs_ach[year] - stud_obs_qual).reshape(-1, 1)
        )[:, 1].reshape((n_colls, n_studs))

        num_apps = 4 + np.rint(0.5 * res[year]).astype(int)

        num_adms = (n_seats // expected_yield).astype(int)

        if year >= 15:  # action
            # racial affirmative action
            aa_race = wgt_aa_race * minority[year]

            # ses affirmative action
            if ses_alt:
                aa_ses = wgt_aa_ses * res[year]
            else:
                if ses_penalty:
                    aa_ses = wgt_aa_ses * -1 * zscore(res[year])
                else:
                    aa_ses = wgt_aa_ses * np.clip(-1 * zscore(res[year]), 0, None)
            coll_obs_ach[year, active] += aa_race + aa_ses

            # targeted recruitment
            util[active[:, np.newaxis] * minority[year, np.newaxis, :]] += wgt_recruit

        # application step
        prev_eu = np.zeros(n_studs)  # expected utility
        for i in range(np.max(num_apps)):
            # expected util of new apps
            eu = util * admit_prob + (1 - admit_prob) * prev_eu
            # best coll for each stud is max eu and not already applied
            best = (eu * ~application[year]).argmax(axis=0)
            # update expected util
            prev_eu = np.take_along_axis(eu, best[None, :], axis=0)
            # apply if enough num apps
            np.put_along_axis(application[year], best[None, :], i < num_apps, axis=0)

        # admission step
        # sort (up to) top applications
        ind_adm = np.argpartition(application[year] * (-coll_obs_ach[year]), num_adms)
        ind_c, ind_s = np.mgrid[:n_colls, :n_studs]
        # select top students
        ind_nadms = ind_s < num_adms[:, np.newaxis]
        ind_admk = ind_c[ind_nadms], ind_adm[ind_nadms]
        admission[year][ind_admk] = application[year][ind_admk]

        # enrollment step
        # max utility if admitted
        ind_enr = np.argmax(util * admission[year], axis=0)
        enr_ = np.take_along_axis(admission[year], ind_enr[None, :], axis=0)
        np.put_along_axis(enrollment[year], ind_enr[None, :], enr_, axis=0)

    # per year and college

    n_minority = (minority[:, np.newaxis, :] * enrollment).sum(axis=2)
    pct_minority = n_minority / enrollment.sum(axis=2)

    mean_ach = (ach[:, np.newaxis, :] * enrollment).sum(axis=2) / enrollment.sum(axis=2)

    # assemble with index

    ind = np.indices((n_years, n_colls, n_studs)).transpose(1, 2, 3, 0)

    ind_yc = ind[:, :, 0, 0], ind[:, :, 0, 1]
    ind_ys = ind[:, 0, :, 0], ind[:, 0, :, 2]
    ind_ycs = ind[..., 0], ind[..., 1], ind[..., 2]

    colleges = np.rec.fromarrays(
        np.broadcast_arrays(*ind_yc, qual, active, pct_minority, mean_ach),
        names=["year", "coll", "qual", "active", "pct_minority", "mean_ach"],
    )
    colleges = pd.DataFrame.from_records(colleges.ravel(), index=["year", "coll"])

    students = np.rec.fromarrays(
        np.broadcast_arrays(*ind_ys, race, res, ach),
        names=["year", "stud", "race", "res", "ach"],
    )
    students = pd.DataFrame.from_records(students.ravel(), index=["year", "stud"])
    students["race"] = pd.Categorical.from_codes(
        students["race"], categories=RACE_NAMES
    )

    outcomes = np.rec.fromarrays(
        np.broadcast_arrays(*ind_ycs, application, admission, enrollment),
        names=["year", "coll", "stud", "application", "admission", "enrollment"],
    )
    outcomes = pd.DataFrame.from_records(
        outcomes.ravel(), index=["year", "coll", "stud"]
    )

    return S(colleges, students, outcomes)
