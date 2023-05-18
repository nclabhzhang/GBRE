# GBRE
This repository provides the implemention for the paper Sentence Bag Graph Formulation for Biomedical Distant Supervision Relation Extraction.

Please cite our paper if our datasets or code are helpful to you ~

## Requirements
* Python 3.6+
* Pytorch 1.4.0+

## Dataset
* Biorel (https://bit.ly/biorel_dataset)
* TBGA (https://zenodo.org/record/5911097)

## Before Training
We need to convert all TBGA dataset to Biorel format.

You can get pretrained word embeddings(biowordvec_mat.npy and biowordvec_word2id.json) of TBGA at (https://github.com/GDAMining/gda-extraction).

Then, run the processing file：

```bash
cd ./data
python process_tbga.py
```


## Training & Evaluation
python train.py

python value.py

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

