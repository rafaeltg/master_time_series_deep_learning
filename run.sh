#!/bin/bash

curr_dir=`pwd`


function update_pydl {
    cd ../Deep-Learning-Algorithms
    git up
    pip3 install -r requirements.txt -U
    sudo python3.5 setup.py install --force -O2
}

function make_dirs {
    rm -rf data inputs models results
    mkdir -p {data,inputs,models,results/{optimize,fit,cv,predict,eval,figs}}
}

function make_inputs {
    make_dirs

    ./create_datasets.py &
    ./create_models.py
    ./create_inputs.py
    wait

    user=`whoami`
    sudo chown -R $user:$user results
}

function install {
    git up
    chmod +x create_datasets.py create_models.py create_inputs.py create_outputs.py

    update_pydl

    cd ../master_time_series_deep_learning

    make_dirs
}

function do_operation {
	for model in "mlp" "sae" "sdae" "lstm"
	do
		for data in "sp500" "mg" "energy"
		do
		    in="$curr_dir"/inputs/"$model"_"$data"_"$1".json
		    out="$curr_dir"/results/"$1"
		    printf "%s - %s - %s\n" $1 $in $out
		    if [ $2 = true ]; then
		        sudo nice -n -19 pydl "$1" -c "$in" -o "$out" &
		    else
		        sudo nice -n -19 pydl "$1" -c "$in" -o "$out"
		    fi
		done
	done
}

function do_op {
    in="$curr_dir"/inputs/"$1"_"$2"_"$3".json
	out="$curr_dir"/results/"$3"
	printf "%s - %s - %s\n" $3 $in $out
	sudo nice -n -19 pydl "$3" -c "$in" -o "$out"
}

OPT=false
CV=false
PRED=false
EVAL=false
OUT=false

while getopts 'uimocpeax' flag; do
  case "${flag}" in
    u) update_pydl ;;
    i) install ;;
    m) make_inputs ;;
    o) OPT=true ;;
    c) CV=true ;;
    p) PRED=true ;;
    e) EVAL=true ;;
    a) OPT=true; CV=true; PRED=true; EVAL=true; OUT=true ;;
    x) ./create_outputs.py ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done


for model in "mlp" "sae" "sdae" "lstm"
do
    printf "\n------\n"

	for data in "sp500" "mg" "energy"
	do
	    do_op "$model" "$data" "optimize"
	    do_op "$model" "$data" "cv" &
	    do_op "$model" "$data" "fit"
	    do_op "$model" "$data" "predict" &
	    do_op "$model" "$data" "eval"
	    wait
	done
done

# OUTPUTS
#if [[ $OUT = true ]] && [[ $PRED = true ]] && [[ $EVAL = true ]]; then
#    ./create_outputs.py
#fi

tar -zcf ../results.tar.gz results/

#if [ $OPT = true ]; then
sudo shutdown -h now
#fi