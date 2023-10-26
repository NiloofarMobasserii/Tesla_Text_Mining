import enum
import pickle
import tqdm
from sklearn.model_selection._split import train_test_split
import pandas as pd
import random


random.seed(7)

df = pd.read_csv("./data/dt-lag.csv")


classes = ["good", "bad", "neutral"]

headlines = list(df["Headline"].values)
articles = list(df["article"].values)


l1s = list(df["Label 1"])
l2s = list(df["Label 2"])
l3s = list(df["Label 3"])
l4s = list(df["Label4(sentiment finbert)"])

multilabels: list = []
for i in range(len(l1s)):
    tmp = list([0, 0, 0])
    tmp[classes.index(l1s[i].lower())] = 1
    tmp[classes.index(l2s[i].lower())] = 1
    tmp[classes.index(l3s[i].lower())] = 1
    tmp[classes.index(l4s[i].lower())] = 1
    multilabels.append(tmp)


texts = [headlines[i] + ". " + articles[i] for i in range(len(headlines))]


train_texts, val_texts, train_labels, val_labels = train_test_split(texts, multilabels, test_size=0.2)

import torch
from transformers import BertTokenizerFast, BertModel


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
model = BertModel.from_pretrained("ProsusAI/finbert", output_hidden_states=True)


model.to(device)
model.eval()

train_encodings = []
val_encodings = []
for t in train_texts:
    train_encodings.append(tokenizer(t, return_tensors="pt", truncation=True, padding=True))
for v in val_texts:
    val_encodings.append(tokenizer(v, return_tensors="pt", truncation=True, padding=True))


t_es: list = []
for i, enc in enumerate(train_encodings):
    if i % 50 == 0:
        print(i)
    encoded = {k: v.to(device) for k, v in enc.items()}
    infered = model(**encoded)
    hidden_states = infered.hidden_states
    token_vectors = hidden_states[-2][0]
    doc_embedding = torch.mean(token_vectors, dim=0)
    embedding = doc_embedding.tolist()
    t_es.append(embedding)
v_es: list = []
for i, enc in enumerate(val_encodings):
    if i % 50 == 0:
        print(i)
    encoded = {k: v.to(device) for k, v in enc.items()}
    infered = model(**encoded)
    hidden_states = infered.hidden_states
    token_vectors = hidden_states[-2][0]
    doc_embedding = torch.mean(token_vectors, dim=0)
    embedding = doc_embedding.tolist()
    v_es.append(embedding)


folder = "./data/ml/lag/"
pickle.dump(t_es, open(folder + "train_encodings", "wb"))

pickle.dump(v_es, open(folder + "val_encodings", "wb"))

pickle.dump(train_labels, open(folder + "train_labels", "wb"))

pickle.dump(val_labels, open(folder + "val_labels", "wb"))
