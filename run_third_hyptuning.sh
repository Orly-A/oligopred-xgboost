#!/bin/sh
#SBATCH --time=96:00:00
#SBATCH --mem=60gb
#SBATCH -c 38

source /vol/ek/Home/orlyl02/working_dir/python3_venv/bin/activate.csh
/vol/ek/Home/orlyl02/working_dir/python3_venv/bin/python3 /vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/third_hyp_param_tuning_multiscore.py > runpicasso_multiscore.log
