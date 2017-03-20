#!/bin/bash

# MODEL NAME in each input = model_data

proj_path='/home/dev/Documents/master/project'
full_path="$proj_path/master_time_series_deep_learning"

function do_operation {
    echo $1
	for model in 'lstm' #'sdae' 'sae' 'mlp'
	do
		for data in 'mg' #'sp500'
		do
			pydl "$1" -c /home/dev/Documents/master/project/master_time_series_deep_learning/inputs/"$model"_"$data"_"$2".json -o $full_path/results/$2/
		done
	done
}

# OPTIMIZATION
do_operation "optimize" "opt"

# CV
#do_operation 'validate' 'cv'

# PREDS
#do_operation 'predict' 'pred'



