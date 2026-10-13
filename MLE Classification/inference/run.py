"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""
import os
import time 
import numpy as np
import pandas as pd
import logging 
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(1, os.getcwd())


from training.train import IrisNetwork, TensorTransformer, preprocessing

import warnings
warnings.filterwarnings('ignore')

# Setting the seed for reproductibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Logging 
logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_file():
   try:
    with open('./model/batch_size.txt', 'r') as f:
       b = f.read()
       f.close
    return int(b)
   except FileNotFoundError:
    logging.info('Error: No batch size file')
    sys.exit(1)
    
def save_results(true, pred):
   if not os.path.isdir('results'):
    os.mkdir('./results')
   rpath = os.path.join(os.getcwd(),'results','results.csv')
   r = pd.DataFrame({
     'Actual': true,
     'Predicted': pred
   })
   r.to_csv(rpath, index = False)
   logging.info(f'Results saved in: {rpath}')

def inference(model, batch, test):
  test_loader = DataLoader(test, batch_size = batch)
  true = test[:,-1].detach().numpy()
  test_loss = 0
  test_acc = 0
  loss_fn = nn.CrossEntropyLoss()
  prediction = np.array([])
  correct = 0
  l = 0
  model.eval()
  s = time.time()
  logging.info(f'Running inference with {batch} batch size and {test.shape[0]} test size.')
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
          prediction = np.concatenate((prediction,pred))

  e = time.time()

  test_loss /= l
  test_acc = correct/l

  logging.info(f"""Inference completed in {e-s} seconds.
               Accuracy score: {test_acc}.""")
  return true, prediction

def main():

    logging.info('Loading data')
    try: 
      X = pd.read_csv('./data_process/test.csv')
    except FileNotFoundError:
       logging.info('Error: No data file')
       sys.exit(1)

    X = preprocessing(X)

    try: 
      y = pd.read_csv('./data_process/ytest.csv')
    except FileNotFoundError:
      logging.info('Error: No data file')
      sys.exit(1)

    y_test = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    testing = torch.cat((X,y_test), 1)

    logging.info('Loading model and batch size')

    try:
      model = IrisNetwork(3,0)
      model = torch.load('./model/model.pt')
    except FileNotFoundError:
      logging.info('Error: No model file')
      sys.exit(1)

    batch_size = load_file()

    true, prediction = inference(model, batch_size, testing)
    save_results(true, prediction)

if __name__ == "__main__":
    main()
