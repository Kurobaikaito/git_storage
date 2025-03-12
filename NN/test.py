import torch
import os
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

breast_cancer = load_breast_cancer()
X = breast_cancer["data"]
y = breast_cancer["target"]

X_tensor = torch.tensor(X,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_tensor,y_tensor,test_size=0.2,random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(X_train.shape)