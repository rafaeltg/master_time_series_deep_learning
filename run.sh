#!/bin/bash

curr_dir=`pwd`

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

    user=`whoami`
    sudo chown -R $user:$user results
}

function do_operation {
	for model in "sdae" "sae" #"mlp" "lstm"
	do
		for data in "sp500" #"mg"
		do
		    in="$curr_dir"/inputs/"$model"_"$data"_"$2".json
		    out="$curr_dir"/results/"$2"
		    echo $1 $2 $3 $in $out
		    if [ $3 = true ]; then
		        sudo nice -n -19 pydl "$1" -c "$in" -o "$out" &
		    else
		        sudo nice -n -19 pydl "$1" -c "$in" -o "$out"
		    fi

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
do_operation "optimize" "opt" false

# CV
do_operation "cv" "cv" true

# PREDS
do_operation "predict" "pred" true

# SCORES
do_operation "eval" "eval" true

wait

tar -zcf results.tar.gz results/

#sudo shutdown -h now