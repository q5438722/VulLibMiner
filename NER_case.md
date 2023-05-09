Here we provide an example of CVE-2020-2146 to show how our NER model works.
 
The description of CVE-2020-2146 is listed as follows:

~~~
Jenkins Mac Plugin 1.1.0 and earlier does not validate SSH host keys when connecting agents created by the plugin, enabling man-in-the-middle attacks.
~~~

It includes the library-name entity: `Jenkins Mac Plugin 1.1.0`, which can be filtered by our NER model.
However, its affected library is `maven:fr.edf.jenkins.plugins:mac`.
The CVE description lacks part of the group id `fr.edf` and reorder the rest of them.

Incorrect order, lack of some componemts, and other difference are mainly because of the complicated structure of Java libraries (including group id and artifact id), and likely to exist in vulnerability descriptions.

Therefore, we use our NER model to address this challenge, and achieves the precision, recall, and F1 score of 96.16%, 92.85%, and 94.48%,respectively.

