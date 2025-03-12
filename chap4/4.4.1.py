from torch.utils.data import Dataset
'''
class Op(object):
    def __init__(self,inputs):
        pass
    
    def __call__(self,inputs):
        return self.forward(inputs)
    
    def forward(self,inputs):
        raise NotImplementedError
    
    def backward(self,inputs):
        raise NotImplementedError
'''
'''
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 加载数据集
iris = load_iris()
X, y = iris.data, iris.target  # 特征和标签

# 2. 数据预处理（标准化 + 划分训练集和测试集）
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 归一化数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 Torch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 3. 创建 DataLoader（批量加载数据）
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# 4. 定义前馈神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # 没有 Softmax，因为交叉熵损失函数包含了 Softmax
        return x


# 5. 初始化模型、损失函数和优化器
input_size = X.shape[1]  # 4 个特征
hidden_size = 10  # 隐藏层神经元数
output_size = len(set(y))  # 3 类
model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器

# 6. 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 7. 模型评估
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = torch.argmax(test_outputs, axis=1)  # 获取最大概率的类别
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"测试集准确率: {accuracy * 100:.2f}%")


from sklearn.datasets import load_iris

# 加载数据集
iris = load_iris()

# 查看数据集的基本信息
print("特征数据（X）:\n", iris.data[:5])  # 只显示前5行数据
print("目标标签（y）:\n", iris.target[:5])
print("特征名称:\n", iris.feature_names)
print("类别名称:\n", iris.target_names)
print("数据集描述:\n", iris.DESCR)  # 数据集详细描述

'''


"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 2. 数据标准化并划分数据集
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 3. 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 4. 定义前馈神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # 没有 softmax，因为交叉熵损失函数已经包含
        return x

# 5. 初始化模型、损失函数和优化器
input_size = X.shape[1]  # 4 个特征
hidden_size = 10
output_size = len(set(y))  # 3 类
model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. 训练模型 + 记录损失值
num_epochs = 50
loss_history = []

for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 7. 绘制训练损失曲线
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', linestyle='-', color='b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练损失曲线")
plt.grid(True)
plt.show()

# 8. 评估模型
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = torch.argmax(test_outputs, axis=1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"测试集准确率: {accuracy * 100:.2f}%")
"""


'''
# 9. 可视化测试集分类结果（使用前两个特征）
plt.figure(figsize=(8, 6))
X_test_plot = X_test[:, :2]  # 只取前两个特征进行可视化
colors = ["r", "g", "b"]
for i in range(3):
    plt.scatter(X_test_plot[predicted == i, 0], X_test_plot[predicted == i, 1],
                color=colors[i], label=f"Predicted Class {i}")

plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.title("测试集分类结果可视化（前两维特征）")
plt.legend()
plt.grid(True)
plt.show()
'''


#from sklearn.datasets import load_breast_cancer


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# 生成数据集：2个特征（方便可视化），3个类别
X, y = make_classification(n_samples=300, n_features=2, n_classes=3, n_clusters_per_class=1, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 直接输出 logits
        return x

# 实例化模型
model = MLP(input_dim=2, hidden_dim=10, output_dim=3)

# 选择损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)  # 前向传播
        loss = criterion(outputs, batch_y)  # 计算损失

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("训练完成！")


# 生成网格点
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

# 预测网格点类别
X_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    Z = model(X_grid)
    Z = torch.argmax(Z, dim=1).numpy()

Z = Z.reshape(xx.shape)

# 画出决策边界
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")  # 颜色填充决策区域
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")  # 绘制样本点
plt.title("三分类神经网络决策边界")
plt.show()

