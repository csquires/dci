#!/usr/bin/env bash

python3 -m scripts.make_dag_pairs --folder fig2_data --npairs 3 --nnodes 100 --nneighbors 10 --changes fixed --percent-added .025 --percent-removed .025
python3 -m scripts.sample_dag_pairs --folder fig1_data --nsamples 300

