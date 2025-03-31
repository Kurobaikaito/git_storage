import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 加载乳腺癌数据集
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0:良性, 1:恶性

# 按类别分离数据
X_benign = X[y == 0]  # 良性数据
X_malignant = X[y == 1]  # 恶性数据

# 计算均值和标准差
mean_benign, std_benign = X_benign.mean(), X_benign.std()
mean_malignant, std_malignant = X_malignant.mean(), X_malignant.std()

# 计算 3σ 范围
lower_benign, upper_benign = mean_benign - 3 * std_benign, mean_benign + 3 * std_benign
lower_malignant, upper_malignant = mean_malignant - 3 * std_malignant, mean_malignant + 3 * std_malignant

# 过滤数据，去除 3σ 之外的异常值
filtered_benign = X_benign[(X_benign >= lower_benign) & (X_benign <= upper_benign)].dropna()
filtered_malignant = X_malignant[(X_malignant >= lower_malignant) & (X_malignant <= upper_malignant)].dropna()

# 重新封装 X 和 y
X_filtered = pd.concat([filtered_benign, filtered_malignant], ignore_index=True)
y_filtered = pd.Series([0] * len(filtered_benign) + [1] * len(filtered_malignant))
X_filtered = X_filtered.to_numpy()
y_filtered = y_filtered.to_numpy()
y_filtered = y_filtered.reshape([-1,1])

X_filtered = torch.tensor(X_filtered,dtype=torch.float32)
y_filtered = torch.tensor(y_filtered,dtype=torch.float32)

X_filtered_train,X_filtered_test,y_filtered_train,y_filtered_test = train_test_split(X_filtered,y_filtered,test_size=0.2,random_state=42)

# 输出数据集信息
print(f"过滤后 X 维度: {X_filtered.shape}")
print(f"过滤后 y 维度: {y_filtered.shape}")
print(f"过滤后 X 维度: {X_filtered_train.shape}")
print(f"过滤后 y 维度: {y_filtered_train.shape}")
#print(y_filtered)
#print(X_filtered.head())  # 查看特征数据
#print(y_filtered.value_counts())  # 查看类别分布






