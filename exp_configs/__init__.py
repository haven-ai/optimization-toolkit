from . import adaptive_exps, sps_exps, benchmark_exps, icml_rebuttal

EXP_GROUPS = {}
EXP_GROUPS.update(adaptive_exps.EXP_GROUPS)
EXP_GROUPS.update(sps_exps.EXP_GROUPS)
EXP_GROUPS.update(benchmark_exps.EXP_GROUPS)
EXP_GROUPS.update(icml_rebuttal.EXP_GROUPS)