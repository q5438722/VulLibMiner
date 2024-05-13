This is our website for `Identifying Vulnerable Third-Party Java Libraries from Textual Descriptions of Vulnerabilities and Libraries`.

In folder `tf-idf`, we include the source code of our TF-IDF matcher. It is invoked by initializing a `search_engine` with a Maven corpus and directly invoking the `search_topk_objects` API.

In folder `trainer_reranking`, we include the source code of our BERT-FNN model, this script can be directly executed by setting the input/outut paths.



In `NER_case.md`, we show an motivating example case of our NER model, illustrating its effectiveness of identifying library-name entities.

We have open-sourced our VulLib and VeraJava dataset in the folder `VulLib` and `VeraJava` respectively.

In `VulLib`, `VulLib/train.json`, `VulLib/valid.json`, `VulLib/test.json` corresponds to the training, validate, and testing set of VulLib.

Additionally, in RQ2, we split the testing set of into the zero-shot testing and the full-shot testing set, we also include them in `VulLib/test_zero.json` and `VulLib/test_full.json`.

In each file, an vulnerability entry includes its CVE id, vulnerability description, labels (affected libraries), Top-128 TF-IDF results and Top-128 results after BERT-FNN (output of VulLibMiner).

Now that we take the descriptions of library descriptions besides vulnerability descriptions as input, `maven_corpus_new.json` corresponds to the descriptions of all Java libraries in Maven (311,233).

An Java libary entry includes its name, description, tokens in its name, tokens in its description.

In `VeraJava`, `VeraJava/cve_labels.csv` and `VeraJava/verajava.csv` corresponds to the original dataset (including multiple programming languages) and the VeraJava dataset we extracted.

In `Chronos`, `FastXML`, `LightXML`, we include our baselines and their running scripts.

In `data_scripts`, we include our scripts for data-cleaning and generation for VulLibMiner and baselines.

In `tf-idf-fig`, we include our evaluation results of RQ4: the selection of hyper-parameters.

