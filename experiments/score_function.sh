#!/usr/bin/env bash

problem=$1
dataset=$2
heatmapname=$3
heatmap="results/${dataset}/heatmaps/${heatmapname}.pkl"

val_size=100

results_dir=results/${dataset}/score_function_experiment
mkdir -p ${results_dir}

echo "Run score function experiments on ${dataset}"

for beam_size in 1 10 100 1000 10000 100000
do
  # See also beam_sizes.sh
  num_processes=1
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
              num_processes=5
          fi
      fi
  fi

  echo "Running beam_size ${beam_size}"
  for score_function in "heatmap_potential" "heatmap" "cost"
  do
    echo "Running beam_size ${beam_size} ${score_function}"
    python eval.py data/${dataset}.pkl \
    --problem ${problem} --offset 0 --decode_strategy dpdp --score_function ${score_function} -f \
    --heatmap ${heatmap} --num_processes ${num_processes} --batch_size ${batch_size} --val_size ${val_size} \
    --beam_size ${beam_size} -o ${results_dir}/beam${beam_size}_${score_function}.pkl

  done
  echo "Running beam_size ${beam_size} plain beam search"
  python eval.py data/${dataset}.pkl \
    --problem ${problem} --offset 0 --decode_strategy dpbs --score_function "heatmap_potential" -f \
    --heatmap ${heatmap} --num_processes ${num_processes} --batch_size ${batch_size} --val_size ${val_size} \
    --beam_size ${beam_size} -o ${results_dir}/beam${beam_size}_dpbs_heatmap_potential.pkl

done

echo "Done"