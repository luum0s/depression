Due to the privacy concerns of the depression dataset, we have applied to the official source of the eRisk2017 dataset for permission to use it. Therefore, we are unable to provide the dataset directly. If you need the dataset, you can also apply to the official source through the following link: https://tec.citius.usc.es/ir/code/dc.html

First, `process_sentence_embedding.py` embeds all the sentences.  

Next, use `make_dataset.ipynb` to produce the processed datasets.

Then, use `know_data` to generate commonsense knowledge, and use `3wd` to generate commonsense knowledge filtered through three-way decision.

If you wish to further filter the knowledge, then run `know_select`.

Finally `bash runme_combine16.sh` can run the experiments of `CoKE`.

Other files:
- data.py : defines the datasets and data module
- model.py : defines the models, including HAN-GRU, BERT and the proposed HAN-BERT (BERTHierClassifierTransAbs)
- main_hier_clf.py : run experiments with HAN-BERT models
- generate_knowledge.py : The tool class used for generating commonsense knowledge.
- utils.py: The tool class used in generate_knowledge.py.