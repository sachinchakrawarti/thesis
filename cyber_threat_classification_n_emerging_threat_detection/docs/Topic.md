






What feature categories does the paper use?
The base paper uses 6 feature families across 2 classifier families:

#	Feature	Type	Handles
1	BOW (binary)	Binary presence (0/1)	Keyword presence
2	BOW (term frequency)	Word counts	Vocabulary frequency
3	TF-IDF (unigram)	Sparse weighted	Keyword importance
4	TF-IDF (bigram)	2-word phrases	Local context
5	Word2Vec (W2V)	Dense 100/300-dim	Word-level semantics
6	Doc2Vec (D2V)	Dense 100/300-dim	Document-level semantics



And the paper's reported numbers (multiclass):

Feature	SVM	XGBoost	Best
BOW binary	96.35%	92.30%	SVM
BOW term frequency	95.35%	91.13%	SVM
TF-IDF unigram	83.06%	86.00%	XGBoost
TF-IDF bigram	75.24%	72.50%	SVM
Word2Vec	58.86%	54.53%	SVM
Doc2Vec	77.46%	66.10%	SVM