import argparse
from itertools import chain

from joblib import Parallel, delayed

from utils_figs import run, kwargs_default_seeded, dict_product

params = {
    "c1": [dict(n_years=50)],
    "c2": [dict(n_years=50, wgt_aa_race=260)],
    "c3": [dict(n_years=50, wgt_aa_race=75, wgt_aa_ses=100)],
    "c4": [dict(n_years=50, wgt_recruit=100, wgt_aa_ses=150)],
    "c4_f1": [dict(n_years=50, wgt_recruit=100, wgt_aa_ses=150)],
    "c4_pen": [dict(n_years=50, wgt_recruit=100, wgt_aa_ses=150, ses_penalty=True)],
    "c4_pq07": [dict(n_years=50, wgt_recruit=100, wgt_aa_ses=150, obs_qual_alt=True)],
    "c5": [dict(n_years=50, wgt_recruit=100, wgt_aa_ses=150, ses_alt=False)],
    "a2": dict_product(dict(wgt_aa_ses=[0, 75, 150], wgt_recruit=[25, 50, 100])),
    "a4": dict_product(dict(wgt_aa_ses=[0, 75, 150], wgt_aa_race=[0, 150, 300])),
    "a3": dict_product(dict(wgt_aa_ses=[0, 75, 150], wgt_recruit=[25, 50, 100])),
    "a5": dict_product(dict(wgt_aa_ses=[0, 75, 150], wgt_aa_race=[0, 150, 300])),
    "2": dict_product(dict(wgt_aa_ses=[0, 50, 100, 150], wgt_recruit=[0, 25, 50, 100]))
         + [dict(wgt_aa_race=260)],
    "d1": [dict(wgt_aa_ses=150, wgt_recruit=100), dict(wgt_aa_race=260)],
    "d2": dict_product(
        dict(wgt_aa_ses=[150], wgt_recruit=[100], n_active=[4, 10, 20, 40])
    ),
    "d3": dict_product(
        dict(wgt_aa_ses=[150], wgt_recruit=[100], n_active=[4, 10, 20, 40])
    ),
}

params = chain.from_iterable(
    kwargs_default_seeded(**kwargs) for p in params.values() for kwargs in p
)

params = (
    dict(kwargs) for kwargs in set(tuple(sorted(kwargs.items())) for kwargs in params)
)

parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jobs", type=int, help="Number of concurrently running jobs. If -1 all CPUs are used. "
                                                   "If below -1, (n_cpus + 1 + n_jobs) are used.", default=1)
parser.add_argument('--verbose', '-v', action='count', default=0)
args = parser.parse_args()

verbose = args.verbose * 4  # -vvv =12 >10 all iterations are reported

_ = Parallel(n_jobs=args.jobs, verbose=verbose)(delayed(run)(**kwargs) for kwargs in params)
