#!/usr/bin/env bash

python3 -m scripts.make_dag_pairs --folder fig2_data --npairs 3 --nnodes 10 --nneighbors 10 --changes fixed --percent-added .025 --percent-removed .025
python3 -m scripts.sample_dag_pairs --folder fig2_data --nsamples 300

matlab -nodesktop -nojvm -r "addpath ./MATLAB_stuff/demo; demo_sparse('data/fig2_data/', 300, 10, [16, 30, 40, 50, 60, 80, 90, 100, 120, 150], 3)"
#python3 -m scripts.run_dci --folder fig2_data --dug kliep --nsamples 300 --alphas .00001 .0001 .001 .01 .1

python3 -m scripts.run_dci --folder fig2_data --nsamples 300 --alphas .00001 .0001 .001 .01 .1
Rscript scripts/run_pcalg.R --folder fig2_data --nsamples 300 --alphas .00001 .0001 .001 .01 .1
Rscript scripts/run_ges.R --folder fig2_data --nsamples 300 --lambdas .1 .3 .5 .7

