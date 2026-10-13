
import os
import logging
import pandas as pd
import time
import sys
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')

# Logging 
logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Setting the seed for reproductibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Custom transformer
class TensorTransformer(BaseEstimator, TransformerMixin):
  def fit(self, X):
    return self
  def transform(self, X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor
  
# Preprocessing pipeline
def preprocessing(data):
  prep = Pipeline([
      ('scaler', StandardScaler()),
      ('tensor', TensorTransformer())
    ])
  data = prep.fit_transform(data)
  return data

# Network class
class IrisNetwork(nn.Module):
  def __init__(self, n_layers, sub):
    super().__init__()
    self.n_layers = n_layers
    self.sub = sub
    self.layers = []
    self.activation = []
    start = 4
    out = 0
    for i in range(1, self.n_layers+1):
      out = start - self.sub
      if i < n_layers:
        self.layers.append(nn.Linear(start, out))
        self.activation.append(nn.ReLU())
        self.add_module(f"layer{i}", self.layers[-1])
        self.add_module(f"act{i}", self.activation[-1])
        start = out
      else:
        self.layers.append(nn.Linear(start, 3))
        self.activation.append(nn.Softmax(dim = 1))
        self.add_module(f"layer{i}", self.layers[-1])
        self.add_module(f"act{i}", self.activation[-1])

  def forward(self, x):
    for i in range(0, self.n_layers):
      x = self.layers[i](x)
      x = self.activation[i](x)
    return x
  
# Test model loop
def test_model(model, batch, test):
  test_loader = DataLoader(test, batch_size = batch, shuffle = True)
  test_loss = 0
  test_acc = 0
  loss_fn = nn.CrossEntropyLoss()
  correct = 0
  l = 0
  model.eval()
  with torch.no_grad():
      for batch in test_loader:
          x = batch[:,:-1]
          y = batch[:,-1]
          y = y.type(torch.LongTensor)
          out = model(x)
          loss = loss_fn(out, y)
          test_loss += loss.item()
          pred = torch.argmax(out, 1).detach().numpy()
          correct += (y.detach().numpy()==pred).sum().item()
          l += batch.shape[0]


  test_loss /= l
  test_acc = correct / l
  return test_loss, test_acc


# Train model loop
def train_model(model, n_epochs, batch_size, lr, train, val):
  
  train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  best_score = 0.0

  for epoch in range(n_epochs):
      for batch in train_loader:
          Xbatch = batch[:,:-1]
          ybatch = batch[:,-1]
          ybatch = ybatch.type(torch.LongTensor)
          y_pred = model(Xbatch)
          loss = loss_fn(y_pred, ybatch)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      _, score = test_model(model, batch_size, val)
      if best_score < score:
          best_score = score
          best_model = deepcopy(model)

  return best_model, best_score

# Saving model 
def save_model(model, batch):
    logging.info('Saving the model')
    if not os.path.isdir('model'): 
       os.mkdir('./model')
    mpath = os.path.join(os.getcwd(),'model','model.pt')
    bpath = os.path.join(os.getcwd(),'model','batch_size.txt')
    torch.save(model, mpath)
    logging.info(f'Model saved in {mpath}')
    with open(bpath,'w') as f:
       f.write(str(batch))
       f.close
    logging.info(f'Batch size saved in {bpath}')

# Tuning machine
def tuning_machine(epochs, batch, lr, n_layers, sub, train, val):

    # Network complexity parameters
    par = []
    nns = []
    for l in n_layers:
        for s in sub:
            par.append({
                'layers': l,
                'sub': s
            })
            a = [4]
            for n in range(1, l):
              a.append(4-n*s)
            a.append(3)
            nns.append(a)

    params = [par[i] for i in range(len(nns)) if min(nns[i])>0]

    # Network training parameters
    params2 = itertools.product(epochs, batch, lr)
    params2 = [list(x) for x in params2]

    # Tuning machine
    best_score = 0

    tune_start = time.time()

    for p in params:
        for x in params2:

            logging.info(f"""Training model with: 
                         {p['layers']} layers
                         {x[0]} epochs
                         {x[1]} batch size
                         {x[2]} learning rate 
                         {train.shape[0]} train size 
                         {val.shape[0]} validation size""") 

            m1 = IrisNetwork(p['layers'], p['sub'])

            s = time.time()
            m2, val_score = train_model(m1, x[0],x[1],x[2], train, val)
            e = time.time()

            logging.info(f'Testing finished in {e-s} seconds')
            logging.info(f'Validation score: {val_score}')

        if val_score > best_score:
            best_score = val_score
            best = deepcopy(m2)
            best_batch = x[1]

    tune_stop = time.time()
    logging.info(f'Time of working machine: {tune_stop - tune_start} seconds')
    logging.info(f'Best accuracy: {best_score}')
    save_model(best, best_batch)


def main():

    lr = [0.001, 0.01, 0.015]
    epochs = [20, 35, 50]
    batch = [6, 10, 15]
    n_layers = [3, 4, 5]
    sub = [-2, -1, 0, 1, 2]

    logging.info('Importing datasets')

    # Import training datasets
    try: 
      X = pd.read_csv('./data_process/train.csv')
    except FileNotFoundError:
      logging.info('Error: No data file')
      os.mkdir('./model')
      sys.exit(1)

    try: 
      y = pd.read_csv('./data_process/ytrain.csv')
    except FileNotFoundError:
      logging.info('Error: No data file')
      os.mkdir('./model')
      sys.exit(1)
    
    logging.info('Creating train and validation sets')

    # Train-validation split and preprocessing
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, shuffle = True)

    X_train = preprocessing(X_train)
    X_val = preprocessing(X_val)

    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    training = torch.cat((X_train,y_train), 1)
    validation = torch.cat((X_val,y_val), 1)

    # Tune network
    logging.info('Tuning machine starts')
    tuning_machine(epochs, batch, lr, n_layers, sub, training, validation)


if __name__ == "__main__":
    main()

    