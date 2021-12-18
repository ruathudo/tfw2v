# TFW2V - A Document Similarity method


## Install:

```bash
pip install -U tfw2v
```

## How to use:

**Given a list of text document in Python List or pandas Series datatype:**
```python
# For example

text = [
    "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.", 
    "Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old.",
    "There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable."
]
```

**We first import and init the model:**

```python
from tfw2v import TFW2V

# init tfw2v model instance
model = TFW2V()
```

**We support to train the word2vec model, or you can pass your own word embedding model based on Gensim library.**
```python
w2v = model.train_w2v(size=100, epochs=5)

# Word embedding model can be saved to the user defined path:
w2v.save("path/to/model")
```

**We support passing a list of stopwords for text processing. Although, this is optional.**
```python
# Example:
stopwords = ["the", "a", "of"]
```

**Now, run the process, the model will train TF-IDF and using pre-trained *w2v* model to enhance the result:**
```python
result = model.run(text, w2v, stopwords, min_tfidf=0.1, lim_token=20, alpha=0.1, lim_most=0.3)
```

**The result is the dictionary with key is the document index, and value is the list of similar doc indexes and similarity score sorted in descending order.**  
  Eg: ``result[0] = [(5, 0.9), (3, 0.85), (8, 0.81), (10, 0.76),...]``.  
  To get the top 10 most similar docs for given ID 7: ``result[7][:10]``

**Given a doc index, we can also get the most similar docs included their text:**
```python
# Eg: the given doc index is 43, we want most 10 similar docs
# It will return the similar docs included their text
sim_docs = model.most_similar(43, k=10)

# output is in pandas Serires format, which can be easily viewed:
sim_docs.head()
# or save:
sim_docs.to_csv("path/to/csv_file.csv")
```

**To save and load the model**
```python
model.save("path/to/tfw2v")
model.load("path/to/tfw2v")
```

## Parameters for model.run() function:
- **w2v**: word embedding model in Gensim datatype. Required.
- **stopwords**: list of stopwords. Optional. Default None.
- **min_tfidf**: min score for accept a token as an important word. Default 0.1.
- **lim_token**: limit number of tokens assumed as important words in case no token meet the min_tfidf score requirement. Default 20.
- **alpha**: the factor to adjust how much information from word2vec will affect the similarity score from tf-idf. Smaller alpha means to expect less impact. Larger alpha means to expect more surprising result. Default 0.1.
- **lim_most**: Given a doc, only re-calculate the ranking for top N percentages of most similar docs. This help the algorithm run faster. It also help to avoid the too surprising result when re-ranking the bottom of the list (least similar docs). Default 1 (all docs). Recommend 0.2 (top 20% docs).

## Development ##

- To build the package, go to the source folder and run:  
  ``python -m build``
- To upload the package to pypi:  
  ``python -m twine upload --repository pypi dist/*``
- Install the new version:  
  ``pip install --no-deps -U tfw2v``


## Cite ##
**This works is on behalf of following paper:**  
Quan Duong, Mika Hämäläinen, and Khalid Alnajjar. (2021). TFW2V: An Enhanced Document Similarity Method for the Morphologically Rich Finnish Language. In the Proceedings of the 1st on Natural Language Processing for Digital Humanities (NLP4DH).

**BibTex:**
```
@inproceedings{duong-etal-2021-tfw2v,
    title = "TFW2V: An Enhanced Document Similarity Method for the Morphologically Rich Finnish Language",
    author = {Duong, Quan  and
      H{\"a}m{\"a}l{\"a}inen, Mika  and
      Alnajjar, Khalid},
    booktitle = "Proceedings of the 1st Workshop on Natural Language Processing for Digital Humanities (NLP4DH)",
    month = dec,
    year = "2021"
}
```