import time
from numpy import average
import torch
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

folder = "./data/pure-l3/"


class MC(nn.Module):
    def __init__(self, output_size: int):
        super(MC, self).__init__()

        self.hidden = nn.Linear(768, 768)
        self.out = nn.Linear(768, output_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, bert_embdding):
        x = self.hidden(bert_embdding)
        x = self.tanh(x)
        x = self.out(x)
        x = self.softmax(x)

        return x


class ClassificationDataset(Dataset):
    """
    Handles batching of data
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


tx = pickle.load(open(folder + "train_encodings", "rb"))
vx = pickle.load(open(folder + "val_encodings", "rb"))
tmp_ty = pickle.load(open(folder + "train_labels", "rb"))
tmp_vy = pickle.load(open(folder + "val_labels", "rb"))

num_train_samples = len(tx)
num_val_samples = len(vx)

ty = []
for i in tmp_ty:
    tmp = list([0, 0, 0])
    tmp[i] = 1
    ty.append(tmp)

vy = []
for i in tmp_vy:
    tmp = list([0, 0, 0])
    tmp[i] = 1
    vy.append(tmp)

trainset = ClassificationDataset(torch.tensor(tx), torch.tensor(ty))
valset = ClassificationDataset(torch.tensor(vx), torch.tensor(vy))

trainloader = DataLoader(trainset, batch_size=16)
valloader = DataLoader(valset, batch_size=16)


device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
Epochs = 10

outstr = "Epoch,TrainLoss,ValLoss,TrainAccuracy,ValAccuracy,TrainF1,ValF1\n"

from torch.optim import AdamW
from torch.nn.functional import binary_cross_entropy
from torch.nn import CrossEntropyLoss
import torch

model = MC(3)
model.to(device)

optimizer = AdamW(model.parameters())
criterion = CrossEntropyLoss()


def calc_f1(y, y_pred, size=3):
    ys = []
    yps = []
    for item in y:
        tmp_y = list([0, 0, 0])
        tmp_y[item] = 1
        ys.append(tmp_y)
    for item in y_pred:
        tmp_yp = list([0, 0, 0])
        tmp_yp[item] = 1
        yps.append(tmp_yp)

    return f1_score(ys, yps, average="weighted")


for epoch in range(Epochs):
    model.train()
    correct = 0.0
    f1 = 0.0
    c = 0
    tloss = 0.0
    for i, (x_batch, y_batch) in enumerate(trainloader):
        x = x_batch.type(torch.FloatTensor)
        y = y_batch.type(torch.FloatTensor)

        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss = binary_cross_entropy(y_pred, y)
        _, predicted = torch.max(y_pred.data, 1)
        _, labels = torch.max(y.data, 1)
        correct += (predicted == labels).sum().item()
        # print(f"Loss at epoch {epoch}, Iter {i} is: {loss}")
        tloss += float(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        f1 += calc_f1(labels.tolist(), predicted.tolist())
        c = i + 1
    tloss = tloss / c
    tf1 = f1 / c
    tacc = (100 * correct) / num_train_samples
    print(f"F1 at Epoch {epoch + 1} is: {f1/c}")
    print(f"Accuracy at Epoch {epoch + 1} is: {(100*correct)/num_train_samples}")
    model.eval()
    val_correct = 0.0
    val_f1 = 0
    c = 0
    vloss = 0.0
    for i, (x_batch, y_batch) in enumerate(valloader):
        x = x_batch.type(torch.FloatTensor)
        y = y_batch.type(torch.FloatTensor)

        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss = binary_cross_entropy(y_pred, y)
        _, predicted = torch.max(y_pred.data, 1)
        _, labels = torch.max(y.data, 1)
        val_correct += (predicted == labels).sum().item()
        val_f1 += calc_f1(labels.tolist(), predicted.tolist())
        vloss += float(loss)
        c = i + 1
        # print(f"Loss at epoch {epoch}, Iter {i} is: {loss}")
    vloss = vloss / c
    vf1 = val_f1 / c
    vacc = (100 * val_correct) / num_val_samples
    print(f"F1 at Epoch {epoch + 1} is: {val_f1/c}")
    print(f"Validation Accuracy at Epoch {epoch + 1} is: {(100*val_correct)/num_val_samples}")
    outstr += f"{epoch+1},{tloss},{vloss},{tacc},{vacc},{tf1},{vf1}"
    outstr += "\n"

with open(folder + "result.csv", "w") as f:
    f.write(outstr)
