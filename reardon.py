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
    num_years=30, num_active=4, wgt_aa_race=0, wgt_aa_ses=0, wgt_recr_race=0, seed=None
):
    rng = np.random.default_rng(seed)

    qual = np.empty((num_years, NUM_COLLS))
    qual[0] = rng.normal(QUALITY_LOC, QUALITY_STD, size=NUM_COLLS)
    qual[0, ::-1].sort()  # sort by (descending) qual

    active = np.zeros(NUM_COLLS, dtype=bool)
    active[:num_active] = True

    expyield = 0.2 + 0.6 * np.argsort(qual[0])[::-1] / (NUM_COLLS - 1)

    race = rng.choice(4, size=(num_years, NUM_STUDS), p=RACE_WEIGHT)
    minority = (race == 1) | (race == 2)  # Black or Hispanic

    res = rng.normal(RES_LOC[race], RES_STD[race])
    # generate ach such that N(ach_loc, ach_std^2) and corr(res, ach) = achres_corr
    b = AR_CORR * ACH_STD / RES_STD
    a = ACH_LOC - b * RES_LOC
    esd = np.sqrt(ACH_STD**2 - b**2 * RES_STD**2)
    ach = a[race] + b[race] * res + esd[race] * rng.normal(size=(num_years, NUM_STUDS))

    sach_rel = np.clip(0.7 + 0.1 * res, 0.5, 0.9)
    sach_err_std = np.sqrt(ach_var * (1 - sach_rel) / sach_rel)
    sach_err = rng.normal(0, sach_err_std, (num_years, NUM_STUDS))
    sach = ach + 0.1 * res + sach_err

    cach_err_std = np.sqrt(ach_var * (1 - 0.8) / 0.8)
    cach_err = rng.normal(0, cach_err_std, (num_years, NUM_COLLS, NUM_STUDS))
    cach = (ach + 0.1 * res)[:, np.newaxis, :] + cach_err

    squal_err_ = rng.normal(size=(num_years, NUM_COLLS, NUM_STUDS))  # pre-draw noise

    application = np.zeros((num_years, NUM_COLLS, NUM_STUDS), dtype=bool)
    admission = np.zeros_like(application)
    enrollment = np.zeros_like(application)

    admit_reg = LogisticRegression(
        warm_start=True, random_state=np.random.RandomState(seed)
    )
    # TODO: -0.015 in the paper but prev reardon used X=Q-A not A-Q?
    admit_reg.coef_ = np.array([[0.015]])
    admit_reg.intercept_ = np.array([0.0])
    admit_reg.classes_ = np.array([0, 1])

    for y in range(num_years):
        # update qual
        if y > 0:
            # mean achievement of last enrolled students
            nenro1 = enrollment[y - 1].sum(axis=1)
            mach = enrollment[y - 1] @ ach[y - 1]
            mach = np.divide(mach, nenro1, out=mach, where=nenro1 != 0)
            qual[y] = 0.9 * qual[y - 1] + 0.1 * mach

        # update expyield
        if y > 0:
            # up to 3 years of adm / enroll
            last3 = slice(max(0, y - 3), y)
            nenro3 = enrollment[last3].sum(axis=(0, 2))
            nadms3 = admission[last3].sum(axis=(0, 2))
            # 0.0001 if 0 enroll
            expyield = np.divide(
                nenro3, nadms3, out=np.full(NUM_COLLS, 0.0001), where=nenro3 != 0
            )

        # update squal
        squal_rel = np.clip(0.7 + 0.1 * res[y], 0.5, 0.9)
        squal_err_std = np.sqrt(
            np.var(qual[y]) * ((1 - squal_rel) / squal_rel)[np.newaxis, :]
        )
        squal_err = squal_err_std * squal_err_[y]  # squal_err_[y] ~ N(0, 1)
        squal = qual[y, :, np.newaxis] + squal_err

        # update admit_reg
        if y > 5:
            last5 = slice(y - 5, y)
            mask = application[last5]
            y_train = admission[last5][mask]
            x_train = ach[last5, np.newaxis, :] - qual[last5, :, np.newaxis]
            x_train = x_train[mask].reshape(-1, 1)
            admit_reg.fit(x_train, y_train)

        util = -250 + squal
        padmit = admit_reg.predict_proba((sach[y] - squal).reshape(-1, 1))[
            :, 1
        ].reshape((NUM_COLLS, NUM_STUDS))
        napps = 4 + np.rint(0.5 * res[y]).astype(int)
        nadms = (NUM_SEATS // expyield).astype(int)

        if y >= 15:  # action
            aa_race = wgt_aa_race * minority[y]
            aa_ses = wgt_aa_ses * np.clip(-1 * zscore(res[y]), 0, None)
            cach[y, active] += aa_race + aa_ses
            util[active[:, np.newaxis] * minority[y, np.newaxis, :]] += wgt_recr_race

        # application step
        prev_eu = np.zeros(NUM_STUDS)
        for i in range(np.max(napps)):
            eu = util * padmit + (1 - padmit) * prev_eu
            best = (eu * ~application[y]).argmax(axis=0)
            prev_eu = np.take_along_axis(eu, best[None, :], axis=0)
            np.put_along_axis(application[y], best[None, :], i < napps, axis=0)

        # admission step
        ind_adm = np.argpartition(application[y] * (-cach[y]), nadms)
        ind_c, ind_s = np.mgrid[:NUM_COLLS, :NUM_STUDS]
        ind_nadms = ind_s < nadms[:, np.newaxis]
        ind_admk = ind_c[ind_nadms], ind_adm[ind_nadms]
        admission[y][ind_admk] = application[y][ind_admk]

        # enrollment step
        ind_enr = np.argmax(util * admission[y], axis=0)
        enr_ = np.take_along_axis(admission[y], ind_enr[None, :], axis=0)
        np.put_along_axis(enrollment[y], ind_enr[None, :], enr_, axis=0)

    num_minority = (minority[:, np.newaxis, :] * enrollment).sum(axis=2)
    pct_minority = num_minority / enrollment.sum(axis=2)

    ach_mean = (ach[:, np.newaxis, :] * enrollment).sum(axis=2) / enrollment.sum(axis=2)

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
