# CAREA
The code and dataset for paper "CAREA: Co-training Attribute and Relation Embeddings for Cross-Lingual Entity Alignment in Knowledge Graphs"

## Citation
If you found the dataset or codes useful, please cite us as:
Baiyang Chen, Xiaoliang Chen, Peng Lu, Yajun Du, "CAREA: Cotraining Attribute and Relation Embeddings for Cross-Lingual Entity Alignment in Knowledge Graphs", Discrete Dynamics in Nature and Society, vol. 2020, Article ID 6831603, 2020. https://doi.org/10.1155/2020/6831603

@article{CAREA,
   author = {Chen, Baiyang and Chen, Xiaoliang and Lu, Peng and Du, Yajun},
   title = {{CAREA}: {C}otraining {A}ttribute and {R}elation {E}mbeddings for {C}ross-{L}ingual {E}ntity {A}lignment in {K}nowledge {G}raphs},
   journal = {Discrete Dynamics in Nature and Society},
   volume = {Vol. 2020},
   pages = {Article ID 6831603},
   ISSN = {1026-0226},
   DOI = {10.1155/2020/6831603},
   url = {https://doi.org/10.1155/2020/6831603},
   year = {2020},
   type = {Journal Article}
}

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
