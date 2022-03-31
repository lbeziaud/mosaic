import numpy as np
from numpy.lib import recfunctions as rfn
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression

NUM_COLLS = 40
NUM_STUDS = 10000

QUALITY_LOC = 1070
QUALITY_STD = 130

NUM_SEATS = 150

RACES = ["Asian", "Black", "Hispanic", "White"]
RACE_WEIGHT = np.array([0.05, 0.15, 0.2, 0.6])

ACH_LOC = np.array([1038, 869, 895, 1052])
ACH_STD = np.array([202, 169, 185, 186])

RES_LOC = np.array([0.012, -0.224, -0.447, 0.198])
RES_STD = np.array([0.833, 0.666, 0.691, 0.657])

AR_CORR = np.array([0.441, 0.305, 0.373, 0.395])

# weighted mixture
ach_loc_ = np.sum(RACE_WEIGHT * ACH_LOC)
ach_var = np.sum(RACE_WEIGHT * (ACH_STD**2 + ACH_LOC**2 - ach_loc_**2))


def run(
    num_years=30,
    num_active=4,
    wgt_aa_race=0,
    wgt_aa_ses=0,
    wgt_recr_race=0,
    seed=None,
    aa_formula1=False,
    aa_penalize=False,
    percept_qual_07=False,
):
    """

    :param num_years:
    :param num_active: number of colleges using policies
    :param wgt_aa_race: weight of race-based afformative action
    :param wgt_aa_ses: weight of SES-based affirmative action
    :param wgt_recr_race: weight of race-based recruiting
    :param seed: seed for random number generator
    :param aa_formula1: alternative SES-based AA
    :param aa_penalize: alternative SES-based AA
    :param percept_qual_07: alternative perception quality
    :return:
    """
    rng = np.random.default_rng(seed)

    qual = np.empty((num_years, NUM_COLLS))
    qual[0] = rng.normal(QUALITY_LOC, QUALITY_STD, size=NUM_COLLS)
    qual[0, ::-1].sort()  # sort by (descending) quality

    # mask is true if college (top num_active) will use policy
    active = np.zeros(NUM_COLLS, dtype=bool)
    active[:num_active] = True

    expected_yield = 0.2 + 0.6 * np.argsort(qual[0])[::-1] / (NUM_COLLS - 1)

    race = rng.choice(4, size=(num_years, NUM_STUDS), p=RACE_WEIGHT)
    minority = (race == 1) | (race == 2)  # Black or Hispanic

    res = rng.normal(RES_LOC[race], RES_STD[race])
    # generate ach such that N(ach_loc, ach_std^2) and corr(res, ach) = achres_corr
    b = AR_CORR * ACH_STD / RES_STD
    a = ACH_LOC - b * RES_LOC
    esd = np.sqrt(ACH_STD**2 - b**2 * RES_STD**2)
    ach = a[race] + b[race] * res + esd[race] * rng.normal(size=(num_years, NUM_STUDS))

    stud_obs_ach_reliability = np.clip(0.7 + 0.1 * res, 0.5, 0.9)
    stud_obs_ach_err_std = np.sqrt(
        ach_var * (1 - stud_obs_ach_reliability) / stud_obs_ach_reliability
    )
    stud_obs_ach_err = rng.normal(0, stud_obs_ach_err_std, (num_years, NUM_STUDS))
    stud_obs_ach = ach + 0.1 * res + stud_obs_ach_err

    coll_obs_ach_err_std = np.sqrt(ach_var * (1 - 0.8) / 0.8)
    coll_obs_ach_err = rng.normal(
        0, coll_obs_ach_err_std, (num_years, NUM_COLLS, NUM_STUDS)
    )
    coll_obs_ach = (ach + 0.1 * res)[:, np.newaxis, :] + coll_obs_ach_err

    stud_obs_qual_err_ = rng.normal(
        size=(num_years, NUM_COLLS, NUM_STUDS)
    )  # pre-draw noise

    application = np.zeros((num_years, NUM_COLLS, NUM_STUDS), dtype=bool)
    admission = np.zeros_like(application)
    enrollment = np.zeros_like(application)

    admit_reg = LogisticRegression(
        warm_start=True, random_state=np.random.RandomState(seed)
    )
    admit_reg.coef_ = np.array([[0.015]])
    admit_reg.intercept_ = np.array([0.0])
    admit_reg.classes_ = np.array([0, 1])

    for y in range(num_years):
        # update quality
        if y > 0:
            # mean achievement of last enrolled students
            num_enroll = enrollment[y - 1].sum(axis=1)
            mean_coll_ach = enrollment[y - 1] @ ach[y - 1]
            mean_coll_ach = np.divide(
                mean_coll_ach, num_enroll, out=mean_coll_ach, where=num_enroll != 0
            )
            qual[y] = 0.9 * qual[y - 1] + 0.1 * mean_coll_ach

        # update expected_yield
        if y > 0:
            # up to 3 years of adm / enroll
            last3 = slice(max(0, y - 3), y)
            num_enroll_3 = enrollment[last3].sum(axis=(0, 2))
            num_admit_3 = admission[last3].sum(axis=(0, 2))
            # 0.0001 if 0 enroll
            expected_yield = np.divide(
                num_enroll_3,
                num_admit_3,
                out=np.full(NUM_COLLS, 0.0001),
                where=num_enroll_3 != 0,
            )

        # update stud_obs_qual
        stud_obs_qual_rel = np.clip(
            0.7 + 0.1 * res[y], 0.5, 0.7 if percept_qual_07 else 0.9
        )
        stud_obs_qual_err_std = np.sqrt(
            np.var(qual[y])
            * ((1 - stud_obs_qual_rel) / stud_obs_qual_rel)[np.newaxis, :]
        )
        stud_obs_qual_err = (
            stud_obs_qual_err_std * stud_obs_qual_err_[y]
        )  # stud_obs_qual_err_[y] ~ N(0, 1)
        stud_obs_qual = qual[y, :, np.newaxis] + stud_obs_qual_err

        # update admit_reg
        if y > 5:
            last5 = slice(y - 5, y)
            mask = application[last5]
            y_train = admission[last5][mask]
            x_train = ach[last5, np.newaxis, :] - qual[last5, :, np.newaxis]
            x_train = x_train[mask].reshape(-1, 1)
            admit_reg.fit(x_train, y_train)

        util = -250 + stud_obs_qual
        admit_prob = admit_reg.predict_proba(
            (stud_obs_ach[y] - stud_obs_qual).reshape(-1, 1)
        )[:, 1].reshape((NUM_COLLS, NUM_STUDS))
        num_apps = 4 + np.rint(0.5 * res[y]).astype(int)
        num_adms = (NUM_SEATS // expected_yield).astype(int)

        if y >= 15:  # action
            aa_race = wgt_aa_race * minority[y]  # racial affirmative action
            # ses affirmative action
            if aa_formula1:
                aa_ses = wgt_aa_ses * res[y]
            else:
                if aa_penalize:
                    aa_ses = wgt_aa_ses * -1 * zscore(res[y])
                else:
                    aa_ses = wgt_aa_ses * np.clip(-1 * zscore(res[y]), 0, None)
            coll_obs_ach[y, active] += aa_race + aa_ses
            util[
                active[:, np.newaxis] * minority[y, np.newaxis, :]
            ] += wgt_recr_race  # targeted recruitment

        # application step
        prev_eu = np.zeros(NUM_STUDS)  # expected utility
        for i in range(np.max(num_apps)):
            eu = (
                util * admit_prob + (1 - admit_prob) * prev_eu
            )  # expected util of new apps
            best = (eu * ~application[y]).argmax(
                axis=0
            )  # best coll for each stud is max eu and not already applied
            prev_eu = np.take_along_axis(
                eu, best[None, :], axis=0
            )  # update expected util
            np.put_along_axis(
                application[y], best[None, :], i < num_apps, axis=0
            )  # apply if enough num apps

        # admission step
        ind_adm = np.argpartition(
            application[y] * (-coll_obs_ach[y]), num_adms
        )  # sort (up to) top applications
        ind_c, ind_s = np.mgrid[:NUM_COLLS, :NUM_STUDS]
        ind_nadms = ind_s < num_adms[:, np.newaxis]  # select top students
        ind_admk = ind_c[ind_nadms], ind_adm[ind_nadms]
        admission[y][ind_admk] = application[y][ind_admk]  # admit

        # enrollment step
        ind_enr = np.argmax(util * admission[y], axis=0)  # max utility if admitted
        enr_ = np.take_along_axis(admission[y], ind_enr[None, :], axis=0)
        np.put_along_axis(enrollment[y], ind_enr[None, :], enr_, axis=0)  # enroll

    # preprocess

    num_minority = (minority[:, np.newaxis, :] * enrollment).sum(
        axis=2
    )  # per year coll
    pct_minority = num_minority / enrollment.sum(axis=2)  # per year coll

    ach_mean = (ach[:, np.newaxis, :] * enrollment).sum(axis=2) / enrollment.sum(
        axis=2
    )  # per year coll

    # format output

    colleges = np.rec.fromarrays(
        np.broadcast_arrays(qual, active, pct_minority, ach_mean),
        names=["qual", "active", "pct_minority", "ach_mean"],
        formats=[np.float16, np.bool_, np.float16, np.float16],
    )

    students = np.rec.fromarrays(
        np.broadcast_arrays(race, res, ach),
        names=["race", "res", "ach"],
        formats=[np.uint8, np.float16, np.float16],
    )

    outcomes = np.rec.fromarrays(
        np.broadcast_arrays(application, admission, enrollment),
        names=["application", "admission", "enrollment"],
        formats=[np.bool_, np.bool_, np.bool_],
    )

    # indices (year, coll, stud) of enrolled
    ind_enr = np.rec.fromarrays(
        np.nonzero(outcomes.enrollment),
        names=["year", "coll", "stud"],
        formats=[np.uint8, np.uint8, np.uint16],
    )

    enrolled = rfn.merge_arrays(
        (
            ind_enr,
            colleges[ind_enr.year, ind_enr.coll],
            students[ind_enr.year, ind_enr.stud],
        ),
        flatten=True,
        asrecarray=True,
    )

    return colleges, students, outcomes, enrolled
