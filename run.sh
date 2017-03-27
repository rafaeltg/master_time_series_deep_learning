#!/bin/bash

mkdir -p {data,inputs,models,results/{opt,cv,pred,eval}}

python3.5 create_datasets.py &
python3.5 create_models.py
python3.5 create_inputs.py
wait

pydl_cli='/home/rafael/Documents/master/project/Deep-Learning-Algorithms/cli'

function do_operation {
	for model in 'lstm' #'sdae' 'sae' 'mlp'
	do
		for data in 'mg' #'sp500'
		do
		    echo "$1" "$model" "$data"
			python3.5 "$pydl_cli"/run.py $1 -c inputs/"$model"_"$data"_"$2".json -o results/$2/
		done
	done
}

# OPTIMIZATION
do_operation 'optimize' 'opt'

# CV
#do_operation 'validate' 'cv'

# PREDS
#do_operation 'predict' 'pred'

# SCORES
#do_operation 'evaluate' 'eval'


#tar -zcf results.tar.gz results/



