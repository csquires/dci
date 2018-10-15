This repository contains the code for the Differential Causal Inference (DCI)
algorithm.

The arXiv preprint is available at [https://arxiv.org/abs/1802.05631]

Authors: Yuhao Wang, Chandler Squires, Anastasiya Belyaeva, and Caroline Uhler.

This project uses Python 3.6.4 and R 3.5.0

To get started, download, unzip, and run:

* `bash make_venv.sh`
* `R -f install_packages.R`

To recreate the figures, run
* `source venv/bin/activate`
* `bash create_fig1ab_data.sh`
* `python3 create_fig1c_data.py`
* `bash create_fig2_data.sh`
* `ipython3 kernel install --user --name=dci`
* `jupyter-notebook`

Then run the notebooks *figure1ab.ipynb*, *figure1c.ipynb*, and *figure2.ipynb*

