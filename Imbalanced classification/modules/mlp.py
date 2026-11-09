import copy
import numpy as np
import torch
import pickle
import tqdm
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class MLPBinaryClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        n_layers=2,
        hidden_size=64,
        dropout=0.0,
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        positive_class_weight=None,
        patience=5,
        threshold=0.5
    ):
        super().__init__() 
        self.epochs = epochs
        self.batch_size = batch_size
        self.positive_class_weight = positive_class_weight
        self.patience = patience
        self.threshold = threshold

        layers = []
        current_size = input_size

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, 1))
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    def train(self, X, y, X_val, y_val):

        # Preprocessing

        num_features = X.columns.difference(['Gender', 'Vehicle_Age', 'Vehicle_Damage','Driving_License','Previously_Insured']).tolist()

        self.preprocessor = ColumnTransformer(
            [
             ('cat', MinMaxScaler(), num_features)
             
             ],
             remainder='passthrough'
        )

        self.X = self.preprocessor.fit_transform(X)
        self.X_val = self.preprocessor.transform(X_val)

        self.X = torch.tensor(np.asarray(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        self.X_val = torch.tensor(np.asarray(self.X_val), dtype=torch.float32)
        self.y_val = torch.tensor(np.asarray(y_val), dtype=torch.float32).reshape(-1, 1)

        if self.positive_class_weight == "balanced":
            n_negative = (self.y == 0).sum()
            n_positive = (self.y == 1).sum()

            if n_positive == 0:
                raise ValueError("No positive class in train set")

            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=n_negative / n_positive
            )
        elif self.positive_class_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    [self.positive_class_weight],
                    dtype=torch.float32,
                )
            )

        if (y_val == 1).sum() == 0:
            raise ValueError("No positive class in validation set")

        loader = DataLoader(
            TensorDataset(self.X, self.y),
            batch_size=self.batch_size,
            shuffle=True,
        )

        best_recall = -1.0
        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        for _ in tqdm(range(self.epochs)):
            self.model.train()

            for X_batch, y_batch in loader:
                logits = self.model(X_batch)
                loss = self.loss_fn(logits, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(self.X_val)
                val_loss = self.loss_fn(val_logits, self.y_val).item()
                val_predictions = (torch.sigmoid(val_logits) >= self.threshold).float()

                true_positives = ((val_predictions == 1) & (self.y_val == 1)).sum()
                actual_positives = (self.y_val == 1).sum()
                try:
                    val_recall = (true_positives / actual_positives).item()
                except ZeroDivisionError:
                    val_recall = 0

            improved = (
                val_recall > best_recall
                or (val_recall == best_recall and val_loss < best_val_loss)
            )

            if improved:
                best_recall = val_recall
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                break

        self.model.load_state_dict(best_weights)
        return best_recall

    def predict_proba(self, X):
        X_tr = self.preprocessor.transform(X)
        X_tr = torch.tensor(np.asarray(X_tr), dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(X_tr))

        return probabilities.numpy().ravel()

    def predict(self, X):

        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)
    
