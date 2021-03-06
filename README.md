# CAREA
The code and datasets for our paper "CAREA: Co-training Attribute and Relation Embeddings for Cross-Lingual Entity Alignment in Knowledge Graphs", a.k.a., CAREA. 

Download this article via: https://doi.org/10.1155/2020/6831603.

## Code
Please use jupyter notebook to run the codes.

### Dependencies

* Python>=3.7
* Keras>=2.2
* Tensorflow>=1.13
* Scipy
* Numpy

## Datasets
The datasets are from [JAPE](https://github.com/nju-websoft/JAPE). Each dataset folder contain below files:

* ent_ids_1: ids for entities in source KG (ZH/JA/FR);
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG (EN);
* training_attrs_1: entity attributes in source KG (ZH/JA/FR);
* training_attrs_2: entity attributes in target KG (EN);
* all_attrs_range: the range code of attributes in source KG (ZH/JA/FR);
* en_all_attrs_range: the range code of attributes in target KG (EN).

## Citation
If you found this model or code useful, please download the citation data from https://doi.org/10.1155/2020/6831603, or cite it as follows:      
```
@article{CAREA,
  author = {Chen, Baiyang and Chen, Xiaoliang and Lu, Peng and Du, Yajun},
  title = {{CAREA}: {C}otraining {A}ttribute and {R}elation {E}mbeddings for {C}ross-{L}ingual {E}ntity {A}lignment in {K}nowledge {G}raphs},
  journal = {Discrete Dynamics in Nature and Society},
  volume = {Vol. 2020},
  pages = {Article ID 6831603},
  ISSN = {1026-0226},
  DOI = {10.1155/2020/6831603},
  url = { https://doi.org/10.1155/2020/6831603 },  
  year = {2020},
 }
```

## Acknowledgement

We refer to the codes of these repos: keras-gat, [MRAEA](https://github.com/MaoXinn/MRAEA). Thanks for their great contributions!
