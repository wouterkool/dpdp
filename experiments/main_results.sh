#!/usr/bin/env bash

problem=$1
dataset=$2
heatmapname=$3
heatmap="results/${dataset}/heatmaps/${heatmapname}.pkl"

val_size=10000

results_dir=results/${dataset}/main_results
mkdir -p ${results_dir}

echo "Run main experiments on ${dataset}"

for beam_size in 10000 100000 1000000
do
  # By default use 2 processes per GPU to use GPU efficiently
	num_processes=1
  batch_size=1
  if [ ${beam_size} -le 10000 ];
		then
		  # For small instances, use a batch
			batch_size=100
		else
			if [ ${beam_size} -le 100000 ];
        then
          num_processes=5
        else
          num_processes=1
      fi
  fi

  echo "Running beam_size ${beam_size}"
  python eval.py data/${dataset}.pkl \
    --problem ${problem} --offset 0 --decode_strategy dpdp --score_function heatmap_potential -f \
    --heatmap ${heatmap} --heatmap_threshold 1e-5 \
    --num_processes ${num_processes} --batch_size ${batch_size} --val_size ${val_size} \
    --beam_size ${beam_size} -o ${results_dir}/beam${beam_size}.pkl
done

echo "Done"