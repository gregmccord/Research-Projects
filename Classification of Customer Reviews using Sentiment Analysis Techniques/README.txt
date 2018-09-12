----------------------------------------------
COS 424: Sentiment Analysis Task (Spring 2018)
Author: Gregory McCord
----------------------------------------------

-featureSelection.py, called as follows
	python featureSelection.py
This program generates an output file containing the data with only the feature
selected terms included. Additionally, output a file containing the new vocabulary.

-testStanfordNLP.py, called as follows
	python testStanfordNLP.py
This program uses the Stanford CoreNLP library, which must be installed by the following
method:
download the software here: https://stanfordnlp.github.io/CoreNLP/download.html
pip install pycorenlp
cd to the stanford-corenlp-full-<date of most recent upload> folder and run the
following command to start the server:
java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

You can now run the testStanfordNLP.py file. Remember to turn off the server by using Ctrl-C.