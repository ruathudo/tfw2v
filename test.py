#%%
import pandas as pd

from tfw2v import TFW2V

#%%
df = pd.read_json('tmp/sample.json')

stopwords = open("tmp/stopwords.txt", "r").read().splitlines()
stopwords = list(stopwords)


model = TFW2V()

#%%
df = df.head(200)

#%%
text = df['title'] + ' ' + df['body']
# %%
w2v = model.train_w2v(text, epochs=5)
# %%
result = model.run(text, w2v, stopwords, min_tfidf=0.1, lim_token=20, alpha=0.1, lim_most=0.3)
# %%
# get most similar docs for given id
similar_docs = model.most_similar(1, k=10)
# %%

# %%


# %%
