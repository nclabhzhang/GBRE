# GBRE
This repository provides the implemention for the paper Sentence Bag Graph Formulation for Biomedical Distant Supervision Relation Extraction.

Please cite our paper if our datasets or code are helpful to you ~

## Requirements
* Python 3.7
* Pytorch 1.12.0
* Transformers 3.0.0

## Dataset
* Biorel (https://bit.ly/biorel_dataset)
* TBGA (https://zenodo.org/record/5911097)

## Before Training
If you want to use TBGA dataset for training GBRE, please convert all the data to Biorel format.

You can get the pretrained word embeddings (biowordvec_mat.npy and biowordvec_word2id.json) of TBGA at https://github.com/GDAMining/gda-extraction.

After making sure all files are in ./data/TBGA, run the processing file：

```bash
cd ./data
python process_tbga.py
```


## Training & Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=0 python value.py
```

## Results
PR curves in our paper are stored in Curves/.

## Data Format
### train.json & dev.json &test.json
```
[
    {
        "head": {
            "CUI": "C0032594",
            "word": "polysaccharide",
            "start": 6,
            "length": 14,
            "split_start": 1
        },
        "tail": {
            "CUI": "C0007289",
            "word": "carrageenin",
            "start": 35,
            "length": 11,
            "split_start": 4
        },
        "sentence": "algal polysaccharide obtained from carrageenin protects 80 to 100 percent of chicken embryos against fatal infections with the lee strain of influenza virus .",
        "relation": "NA",
        "lexical_feature0": "algal|polysaccharide|obtained|from|carrageenin|protects",
        "lexical_feature1": "PAD|algal|polysaccharide|obtained|from|carrageenin|protects|80",
        "lexical_feature2": "PAD|PAD|algal|polysaccharide|obtained|from|carrageenin|protects|80|to",
        "syntactic_feature0": "polysaccharide|0|root",
        "syntactic_feature1": "obtained|1|acl",
        "syntactic_feature2": "from|4|case",
        "syntactic_feature3": "carrageenin|2|obl"
    },
    ...
]
```


### relation2id.json
```
{
    "NA": 0,
    "active_metabolites_of": 1,
    ...
}
```


### word2vec.json
```
[
    {"word": "a", "vec": [0.0023810673, 0.23303464, ...]},
    {"word": "monocyte", "vec": [0.13468756, -0.18540461, ...]},
    ...
]
```

## Acknowledgements
GBRE builds upon the source code from the project [CGRE](https://github.com/tmliang/CGRE).

We thank their contributors and maintainers!



## Cite

If you use or extend our work, please cite the following:

```
@article{gbre-2023,
  title = "Sentence Bag Graph Formulation for Biomedical Distant Supervision Relation Extraction",
  author = "H.Zhang, Y. Liu, X. Liu, T. Liang, G. Sharma, L. Xue, and M. Guo",
  year = "2023",
}
```
