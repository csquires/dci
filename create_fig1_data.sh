#!/usr/bin/env bash

python3 -m scripts.make_dag_pairs --folder fig1_data --npairs 100 --nnodes 10 --nneighbors 3 --changes independent --percent-added .1 --percent-removed .1
python3 -m scripts.sample_dag_pairs --folder fig1_data --nsamples 1000
python3 -m scripts.sample_dag_pairs --folder fig1_data --nsamples 10000

python3 -m scripts.run_dci --folder fig1_data --nsamples 1000 --alphas .00001 .0001 .001 .01 .1
python3 -m scripts.run_dci --folder fig1_data --nsamples 10000 --alphas .00001 .0001 .001 .01 .1

Rscript scripts/run_pcalg.R --folder fig1_data --nsamples 1000 --alphas .00001 .0001 .001 .01 .1
Rscript scripts/run_pcalg.R --folder fig1_data --nsamples 10000 --alphas .00001 .0001 .001 .01 .1

Rscript scripts/run_ges.R --folder fig1_data --nsamples 1000
Rscript scripts/run_ges.R --folder fig1_data --nsamples 10000