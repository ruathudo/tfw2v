# TFW2V - A Document Similarity method


## Usages:

```python
from tfw2v import TFW2V



# The input is list of text documents in Python List or pandas Series datatype.
# For example
text = [
    "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.", 
    "Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old.",
    "There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable."
]

# We support passing a list of stopwords for text processing
stopwords = ["the", "a", "of"]

# init tfw2v model instance
model = TFW2V()

# We support to train the word2vec model 
# or pass your own word2vec model based Gensim library

w2v = model.train_w2v(size=100, epochs=5)

# now, run the process, the model will train TFIDF and using pre-trained w2v to enhence the result

result = model.run(text, w2v, stopwords, min_tfidf=0.1, lim_tk=20, alpha=0.1, lim_most=0.3)

# the result is the dictionary with key is the document index, and value is the list of similar doc indexes sorted desc.
# Eg: result[0] = [5, 3, 8, 10....]
# Given a doc index, we can get the most similar docs included their text:
# Eg: the given doc index is 43, we want most 10 similar docs
sim_docs = model.most_similar(43, k=10)


```