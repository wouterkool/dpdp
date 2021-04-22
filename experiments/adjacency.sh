#!/usr/bin/env bash

problem=$1
dataset=$2
heatmapname=$3
heatmap="results/${dataset}/heatmaps/${heatmapname}.pkl"

val_size=100
beam_size=100000
batch_size=1
# Five processes per GPU
num_processes=5

echo "Run adjacency experiments on ${dataset}"
results_dir=results/${dataset}/adj_experiment
mkdir -p ${results_dir}

for th in 1e-5 1e-4 1e-3 1e-2 0.1 0.2 0.5 0.9
do
  echo "Run threshold ${th}"
  python eval.py data/${dataset}.pkl --problem ${problem} --offset 0 --decode_strategy dpdp --score_function heatmap_potential -f \
 --heatmap ${heatmap} --heatmap_threshold ${th} --num_processes ${num_processes} --batch_size ${batch_size} \
  --val_size ${val_size} --beam_size ${beam_size} -o ${results_dir}/heatmapth${th}.pkl
done

for knn in 5 10 20 50 99
do
  echo "Running knn ${knn}"
  python eval.py data/${dataset}.pkl --problem ${problem} --offset 0 --decode_strategy dpdp --score_function heatmap_potential -f \
 --heatmap ${heatmap} --knn ${knn} --num_processes ${num_processes} --batch_size ${batch_size} \
 --val_size ${val_size} --beam_size ${beam_size} -o ${results_dir}/knn${knn}.pkl
done

echo "Running fully connected"
python eval.py data/${dataset}.pkl --problem ${problem} --offset 0 --decode_strategy dpdp --score_function heatmap_potential -f \
 --heatmap ${heatmap} --num_processes ${num_processes} --batch_size ${batch_size} \
 --val_size ${val_size} --beam_size ${beam_size} -o ${results_dir}/full.pkl
echo "Done"