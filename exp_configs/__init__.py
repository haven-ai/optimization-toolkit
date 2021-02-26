from . import adaptive_exps, sps_exps, semseg_exps

EXP_GROUPS = {}
EXP_GROUPS.update(adaptive_exps.EXP_GROUPS)
EXP_GROUPS.update(sps_exps.EXP_GROUPS)
EXP_GROUPS.update(semseg_exps.EXP_GROUPS)
