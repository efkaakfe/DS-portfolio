import unittest
import pandas as pd
import os
import numpy as np
import torch
import glob
import sys

sys.path.insert(1, os.getcwd())


from training.train import IrisNetwork, train_model, test_model
from inference.run import inference

# Data test
class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = 'data_process'

    def test_data_read(self):
        os.chdir(self.data_dir)
        for file in glob.glob('*.csv'):
            df = pd.read_csv(file)
            self.assertEqual(df.empty, False)
    def test_sizes(self):

        d1 = pd.read_csv('train.csv')
        d2 = pd.read_csv('ytrain.csv')
        d3 = pd.read_csv('test.csv')
        d4 = pd.read_csv('ytest.csv')

        self.assertEqual(d1.shape[0], d2.shape[0])
        self.assertEqual(d3.shape[0], d4.shape[0])
        
# Network test
class TestNetwork(unittest.TestCase):
    def test_network(self):
        n = IrisNetwork(3,0)
        data = torch.tensor(pd.DataFrame({
            'X1': np.random.rand(30),
            'X2': np.random.rand(30),
            'X3': np.random.rand(30),
            'X4': np.random.rand(30),
        }).to_numpy(), dtype=torch.float32)
        output = n(data)
        
        self.assertEqual(output.shape[1], 3)

# Training, inference codes tests
class TestScripts(unittest.TestCase):
    def test_train_weights(self):
        n = IrisNetwork(3,0)
        t = torch.tensor(pd.DataFrame({
            'X1': np.random.rand(100),
            'X2': np.random.rand(100),
            'X3': np.random.rand(100),
            'X4': np.random.rand(100),
            'y': np.random.randint(0, 3, 100)
        }).to_numpy(), dtype=torch.float32)
        v = torch.tensor(pd.DataFrame({
            'X1': np.random.rand(40),
            'X2': np.random.rand(40),
            'X3': np.random.rand(40),
            'X4': np.random.rand(40),
            'y': np.random.randint(0, 3, 40)
        }).to_numpy(), dtype=torch.float32)

        d1 = n.state_dict()
        w11, w12, w13 = d1['layer1.weight'], d1['layer2.weight'], d1['layer3.weight']

        m, s = train_model(n, 5, 20, 0.001, t, v)
        d2 = m.state_dict()
        w21, w22, w23 = d2['layer1.weight'], d2['layer2.weight'], d2['layer3.weight']

        self.assertEqual(torch.all(w11.eq(w21)), torch.tensor(False))
        self.assertEqual(torch.all(w12.eq(w22)), torch.tensor(False))
        self.assertEqual(torch.all(w13.eq(w23)), torch.tensor(False))

    def test_test(self): 
        n = IrisNetwork(3,0)
        t = torch.tensor(pd.DataFrame({
            'X1': np.random.rand(40),
            'X2': np.random.rand(40),
            'X3': np.random.rand(40),
            'X4': np.random.rand(40),
            'y': np.random.randint(0, 3, 40)
        }).to_numpy(), dtype=torch.float32)
        l, s = test_model(n, 20, t)

        self.assertTrue(0 <= s <=1)

    def test_infer(self): 
        n = IrisNetwork(3,0)
        t = torch.tensor(pd.DataFrame({
            'X1': np.random.rand(40),
            'X2': np.random.rand(40),
            'X3': np.random.rand(40),
            'X4': np.random.rand(40),
            'y': np.random.randint(0, 3, 40)
        }).to_numpy(), dtype=torch.float32)

        true, prediction = inference(n, 20, t)

        self.assertIsInstance(true, np.ndarray)
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(all(np.unique(true)), all(np.unique(prediction)))
        self.assertEqual(true.shape[0], prediction.shape[0])

if __name__ == '__main__':
    unittest.main()

