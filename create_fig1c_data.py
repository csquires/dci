import os

for v in range(10):
    os.system('python3 -m scripts.make_dag_pairs --folder fig1c_data/v%d --npairs 100 --nnodes 10 --nneighbors 3 '
              '--changes independent --percent-added .05 --percent-removed .05 --changed-variances %d' % (v, v))
for v in range(10):
    os.system('python3 -m scripts.sample_dag_pairs --folder fig1c_data/v%d --nsamples 1000' % v)
    os.system('python3 -m scripts.sample_dag_pairs --folder fig1c_data/v%d --nsamples 10000' % v)
for v in range(10):
    os.system('python3 -m scripts.run_dci --folder fig1c_data/v%d --nsamples 1000 --alphas .05' % v)
    os.system('python3 -m scripts.run_dci --folder fig1c_data/v%d --nsamples 10000 --alphas .05' % v)

