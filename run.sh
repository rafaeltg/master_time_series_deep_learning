#!/bin/bash

# MODEL NAME in each input = model_data

function do_operation {
    echo $1
	for model in 'lstm' #'sdae' 'sae' 'mlp'
	do
		for data in 'mg' #'sp500'
		do
			python3.5 ../run.py $1 -c inputs/"$model"_"$data"_"$2".json -o results/$2/	
		done
	done
}

# OPTIMIZATION
do_operation "optimize" "opt"

# CV
#do_operation 'validate' 'cv'

# PREDS
#do_operation 'predict' 'pred'



