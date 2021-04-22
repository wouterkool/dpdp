#!/usr/bin/env bash

problem=$1
dataset=$2
heatmapname=$3
heatmap="results/${dataset}/heatmaps/${heatmapname}.pkl"

val_size=100

echo "Run beam size experiments on ${dataset}"
results_dir=results/${dataset}/beam_size_experiment
mkdir -p ${results_dir}

for beam_size in 10000 25000 50000 100000 250000 500000 1000000 2500000
do
  # For small beam sizes, we cannot spawn enough processes to saturate GPU and it is better to use batching
  num_processes=1
  # Batch size of 1 works best for large beam sizes as
  # there is no overhead for batching and slightly more efficient bounding
  # but we'll use more processes to saturate GPU
  batch_size=1
  if [ ${beam_size} -le 10 ];
    then
      batch_size=10000
    else
      if [ ${beam_size} -le 1000 ];
        then
          batch_size=1000
        else
          if [ ${beam_size} -le 10000 ];
            then
              batch_size=100
            else
              if [ ${beam_size} -le 100000 ];
                then
                  num_processes=5
                else
                  if [ ${beam_size} -le 500000 ];
                    then
                      num_processes=2
                    else
                      num_processes=1
                  fi
              fi
          fi
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