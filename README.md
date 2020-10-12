# CAREA
The code and dataset for paper "CAREA: Co-training Attribute and Relation Embeddings for Cross-Lingual Entity Alignment in Knowledge Graphs"

## Datasets

The datasets are from [BootEA](https://github.com/nju-websoft/BootEA)

* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;
* rel_ids_1: ids for entities in source KG;
* rel_ids_2: ids for entities in target KG;

## Environment

* Python>=3.7
* Keras>=2.2
* Tensorflow>=1.13
* Scipy
* Numpy

## Acknowledgement

We refer to the codes of these repos: keras-gat, [MRAEA](https://github.com/MaoXinn/MRAEA). Thanks for their great contributions!
