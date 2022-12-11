# Effective Higher-order Link Prediction and Reconstruction from Simplicial Complex Embeddings

In this repository you find code to run experiments of the paper "Effective Higher-order Link Prediction and Reconstruction from Simplicial Complex Embeddings". The library [SNAP](https://github.com/snap-stanford/snap) is needed, together with `requirements.txt`. The code has been tested with Python 3.6.13.

If you use the code in this repository, please cite us:
```bibtex
@inproceedings{
piaggesi2022effective,
title={Effective Higher-order Link Prediction and Reconstruction from Simplicial Complex Embeddings},
author={Simone Piaggesi and Andr{\'e} Panisson and Giovanni Petri},
booktitle={The First Learning on Graphs Conference},
year={2022},
url={https://openreview.net/forum?id=UiBiLRXR0G}
}
```

## Repository organization

### Description of folders

- [processed-data/](processed-data/): datasets downloaded from [here](https://github.com/arbenson/ScHoLP-Data) and pre-processed (extraction of the largest projected component and filtering of unfrequent nodes). Original data can be downloaded also from [here](https://drive.google.com/file/d/1zyonCNnoP7b5Kh7Kq7OSfeI6opvWSckV/view?usp=share_link).
- [3way-metrics-data/](https://drive.google.com/file/d/1losF2t22v7RZhi9hmLASvPKtAo7nUT02/view?usp=share_link): 3-way scores computed with this [code](https://github.com/arbenson/ScHoLP-Tutorial). We slightly adjusted the Julia code to calculate scores also for the quantiles 0-80 needed in the reconstruction task. You need to download this folder to run the 3-way analysis.

### Description of python files

-`simplex2hasse.py`
File got from the reference [repo](https://github.com/lordgrilo/Simplex2Vec) with few modifications for our purposes.

-`snap_node2vec.py`
Utilities to run node2vec with the SNAP library.

-`k_simple2vec.py`
Utilities to run *k-simplex2vec*, downloaded from the reference [repo](https://github.com/celiahacker/k-simplex2vec).

-`calibrated_metrics.py`
Utilities to compute calibrated AUC-PR, downloaded from the reference [repo](https://github.com/wissam-sib/calibrated_metrics).

-`utils.py`
Utilities for our experiments.


### Description of jupyter notebooks

-`s2v_train.ipynb` and `ks2v_train.ipynb`
Jupyter notebooks to train Simplex2Vec and K-Simplex2Vec.

-`negative_hyperedge_sampling.ipynb`
Jupyter notebook to sample negative hyperedges. You need to set the variable TUPLE_SIZE=3 (or 4) to sample results.

-`s2v_dimensions_gridsearch.ipynb` and *ks2v_dimensions_gridsearch.ipynb*
Jupyter notebooks needed to find the best dimension scores in order to plot figures.

-`3way_figures.ipynb` and `3way_figures.ipynb`
Plot figures with previously computed results.

-`4way_tables.ipynb` and `4way_tables.ipynb`
Show results into tables.


## References
1. Billings, Jacob Charles Wright, et al. "Simplex2vec embeddings for community detection in simplicial complexes." arXiv preprint arXiv:1906.09068 (2019).
2. Benson, Austin R., et al. "Simplicial closure and higher-order link prediction." Proceedings of the National Academy of Sciences 115.48 (2018): E11221-E11230.
3. Hacker, Celia. "k-simplex2vec: a simplicial extension of node2vec." arXiv preprint arXiv:2010.05636 (2020).
4. Siblini, Wissam, et al. "Master your metrics with calibration." International Symposium on Intelligent Data Analysis. Springer, Cham, 2020.
