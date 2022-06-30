MSc thesis on Embeddings store monitoring
==========
[![DOI](https://zenodo.org/badge/DOI/110.5281/zenodo.5625651.svg)](https://doi.org/10.5281/zenodo.5625651)

My MSc thesis on Embeddings Monitoring

![](https://raw.githubusercontent.com/theofpa/embeddings-store-thesis/main/images/compacte-logo.jpg)

A Dissertation submitted in partial fulfillment of the requirements for the degree of Master of Science.

### Abstract

Language models produce multidimensional vectors, but the context of the embedding is not known.
Storing the embeddings in a platform also requires advanced monitoring capabilities to detect data shifts.
Drift detection in the domain of contextual embeddings is a challenging task, as it requires a large number of data points to be compared.
This thesis is an empirical study of the problem of detecting data shifts in the domain of contextual embeddings.
We see that a pre-trained classifier for dimensionality reduction can be used to detect data shifts, and performs better than traditional methods like MMD, KS and LSDD.
We also analyse the problem of scaling such a solution and discuss the implications of implementing such a system in a ML platform.

### Reproduce experiments
The experiments were conducted using a i9-12900KF CPU and a RTX3090 GPU on Ubuntu Linux 22.04 with Python 3.8.

```bash
cd experiments
pip install -r requirements.txt
./run.sh
```

# Sacred
```
omniboard --mu "mongodb+srv://xxx:xxx@xxx.otmss.mongodb.net/xxx?retryWrites=true"
```
