#!/bin/bash

mkdir -p {data,inputs,models,results/{opt,cv,pred}}

python3.5 create_datasets.py &
python3.5 create_models.py
python3.5 create_inputs.py
wait


function do_operation {
	for model in 'lstm' 'sdae' 'sae' 'mlp'
	do
		for data in 'mg' 'sp500'
		do
			python3.5 run.py $1 -c inputs/"$model"_"$data"_"$2".json -o results/$2/
		done
	done
}

# OPTIMIZATION
do_operation 'optimize' 'opt'

# CV
do_operation 'validate' 'cv'

# PREDS
do_operation 'predict' 'pred'


tar -zcf results.tar.gz results/



