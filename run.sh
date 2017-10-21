#!/bin/bash

curr_dir=`pwd`
datasets=()
models=()
shutdown=false


function update_pydl {
    cd ../pydl
    git up
    pip3 install -r requirements.txt -U
    sudo python3.5 setup.py install --force -O2
}

function update_master {
    git up
    chmod +x create_inputs.py create_outputs.py

    rm -rf data inputs models results
    mkdir -p {data,inputs,models,results/{optimize,fit,cv,predict,eval,figs,desc}}

    user=`whoami`
    sudo chown -R $user:$user results
}


function do_op {
    in="$curr_dir"/inputs/"$1"_"$2"_"$3".json
	out="$curr_dir"/results/"$3"
	printf "## %s \n" $3
    pydl "$3" -c "$in" -o "$out"
}

function run {

    echo "Creating inputs..."
    ./create_inputs.py --models ${models[*]} --datasets ${datasets[*]}

    for m in ${models[*]}
    do
        for ds in ${datasets[*]}
        do
            printf "\n>> %s - %s ------ %s\n\n" "$ds" "$m" "$(date +'%d/%m/%Y %H:%m:%S')"

            do_op "$m" "$ds" "optimize"
            do_op "$m" "$ds" "cv" &
            do_op "$m" "$ds" "fit"
            do_op "$m" "$ds" "predict"
            do_op "$m" "$ds" "eval"
    	    wait
        done
    done

    echo "Creating outputs..."
	./create_outputs.py --models ${models[*]} --datasets ${datasets[*]}

	tar -zcf ../results.tar.gz results/
}


while getopts 'pud:m:os' flag; do
  case "${flag}" in
    p) update_pydl ;;
    u) update_master ;;
    d) datasets=( $(IFS=" " echo "$OPTARG") ) ;;
    m) models=( $(IFS=" " echo "$OPTARG") ) ;;
    s) shutdown=true ;;
    *) error "Unexpected option ${flag}" ;;
  esac
done


if [ -z "$datasets" ]; then
    echo "No dataset selected"
    exit 1
fi

if [ -z "$models" ]; then
    echo "No models selected"
    exit 1
fi

run

if [ "$shutdown" = true ]; then
    sudo shutdown -h now
fi