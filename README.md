This is our website for `Identifying Vulnerable Third-Party Java Libraries from Textual Descriptions of Vulnerabilities and Libraries`.

In `NER_case.md`, we show an motivating example case of our NER model, illustrating its effectiveness of identifying library-name entities.

We have open-sourced our VulLib and VeraJava dataset in the folder `VulLib` and `VeraJava` respectively.

In `VulLib`, `VulLib/train.json`, `VulLib/valid.json`, `VulLib/test.json` corresponds to the training, validate, and testing set of VulLib.

Additionally, in RQ2, we split the testing set of into the zero-shot testing and the full-shot testing set, we also include them in `VulLib/test_zero.json` and `VulLib/test_full.json`.

In each file, an vulnerability entry includes its CVE id, vulnerability description, labels (affected libraries), Top-128 TF-IDF results and Top-128 results after BERT-FNN (output of VulLibMiner).

Now that we take the descriptions of library descriptions besides vulnerability descriptions as input, `VulLib/maven_corpus.zip` corresponds to the descriptions of all Java libraries in Maven (310,844).

An Java libary entry includes its name, description, tokens in its name, tokens in its description.

In `VeraJava`, `VeraJava/cve_labels.csv` and `VeraJava/verajava.csv` corresponds to the original dataset (including multiple programming languages) and the VeraJava dataset we extracted.

In `Chronos`, `FastXML`, `LightXML`, we include our baselines and scripts.

In `data_scripts`, we include our scripts for data-cleaning and generation for VulLibMiner and baselines.

We do not open-source our code repository now due to the security requirement of our industry partner, and we plan to open-source them at the publication of our paper.
However, in Section 3, we list all details of our approach, so that it can be easily reproduced.

