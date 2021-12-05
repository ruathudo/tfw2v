# TFW2V - A Document Similarity method


## Install:

```bash
pip install tfw2v
```

## How to use:

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
# Although, this is optional.
stopwords = ["the", "a", "of"]

# init tfw2v model instance
model = TFW2V()

# We support to train the word2vec model
# or pass your own word embedding model based Gensim library

w2v = model.train_w2v(size=100, epochs=5)

# Word embedding model can be saved to the user defined path:
w2v.save("path/to/model")

# now, run the process, the model will train TFIDF and using pre-trained w2v to enhence the result

result = model.run(text, w2v, stopwords, min_tfidf=0.1, lim_token=20, alpha=0.1, lim_most=0.3)

# the result is the dictionary with key is the document index, and value is the list of similar doc indexes and similarity score sorted desc.
# Eg: result[0] = [(5, 0.9), (3, 0.85), (8, 0.81), (10, 0.76),...]

# To get the top 10 most similar docs for given ID 7: result[7][:10]

# Given a doc index, we can also get the most similar docs included their text:
# Eg: the given doc index is 43, we want most 10 similar docs
# It will return the similar docs included their text
sim_docs = model.most_similar(43, k=10)

# output is in pandas Serires format, which can be easily viewed:
sim_docs.head()
# or save:
sim_docs.to_csv("path/to/csv_file.csv")


```

## Parameters:
- **min_tfidf**: min score for accept a token as an important word. Default 0.1.
- **lim_token**: limit number of tokens assumed as important words in case no token meet the min_tfidf score requirement. Default 20.
- **alpha**: the factor to adjust how much information from word2vec will affect the similarity score from tf-idf. Smaller alpha means to expect less impact. Larger alpha means to expect more surprising result. Default 0.1.
- **lim_most**: Given a doc, only re-calculate the ranking for top N percentages of most similar docs. This help the algorithm run faster. It also help to avoid the too surprising result when re-ranking the bottom of the list (least similar docs). Default 1 (all docs). Recommend 0.2 (top 20% docs).

** Development

- To build the package, go to the source folder and run:  
  ``python -m build``
- To upload the package to pypi:  
  ``python -m twine upload --repository pypi dist/*``
- Install the new version:  
  ``pip install --no-deps -U tfw2v``