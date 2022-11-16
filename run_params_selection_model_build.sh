#!/bin/sh
#SBATCH --time=96:00:00
#SBATCH --mem=40gb
#SBATCH -c 20

source /vol/ek/Home/orlyl02/working_dir/python3_venv/bin/activate.csh
/vol/ek/Home/orlyl02/working_dir/python3_venv/bin/python3 /vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/parse_calc_best_params_xgboost.py > calc_best_params.log
