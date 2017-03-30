#!/bin/bash

function update {
    cd ../Deep-Learning-Algorithms
    git up
    pip3 install -r requirements.txt -U
    python3.5 setup.py install --force -O2
    cd ../master_time_series_deep_learning
}

function make_inputs {
    rm -rf data inputs models
    mkdir -p {data,inputs,models,results/{opt,cv,pred,eval}}

    python3.5 create_datasets.py &
    python3.5 create_models.py
    python3.5 create_inputs.py
    wait
}

function do_operation {
	for model in "lstm" #"sdae" "sae" "mlp"
	do
		for data in "sp500" "mg"
		do
		    echo $1 $model $data
		    sudo nice -n -20 pydl $1 -c inputs/"$model"_"$data"_"$2".json -o results/$2/
		done
	done
}


while getopts 'ui' flag; do
  case "${flag}" in
    u) update ;;
    i) make_inputs ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done

# OPTIMIZATION
do_operation "optimize" "opt" 0

# CV
do_operation "cv" "cv" 0

# PREDS
do_operation "predict" "pred" 1

# SCORES
do_operation "evaluate" "eval" 1

tar -zcf results.tar.gz results/


sudo shutdown -h now