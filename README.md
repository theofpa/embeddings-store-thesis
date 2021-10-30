MSc thesis on Embeddings store monitoring
==========
[![DOI](https://zenodo.org/badge/DOI/110.5281/zenodo.5625651.svg)](https://doi.org/10.5281/zenodo.5625651)

My MSc thesis on Embeddings Monitoring

![uva logo](https://raw.githubusercontent.com/theofpa/embeddings-store-thesis/main/images/compacte-logo.jpg | width=200)

A Dissertation submitted in partial fulfillment of the requirements for the degree of Master of Science.

Abstract
==========

Embeddings are taking the AI world by storm. Word embeddings enabled Google’s BERT model, embeddings lie behind Google Translate and other ad- vances in NLP, while graph embeddings have enabled breakthroughs in fighting financial crime. Embeddings are challenging to operationalize. If you change how they are computed, you need a new version of them. In a feature store, this means you may need to retrain all training datasets that use that embedding as a feature. Embeddings may also be published to an embeddings store for similarity search (find me the closest items to ‘X’). In this thesis, I will work on adding MLOps support for computing embeddings, and adding orchestration support in Hopsworks for automating the re-computation of derived features and training datasets when embeddings are re-computed.
