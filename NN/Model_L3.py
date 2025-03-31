import torch
import os
import numpy as np
import pandas as pd
from numpy.random import logistic
from sympy import false
from sympy.printing.pretty.pretty_symbology import line_width
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from test import X_filtered, y_filtered

# 需要正则化的标签
regularization = []

# 存储预测错误的数据
mis_predict = []
X_mis = []
y_mis = []

# 存储预测值在0.4至0.6之间的数据
__sensitive__ = []
_logit_ = []
X_critical = []
y_critical = []
loss_grad_character = None

def reset():
    global mis_predict,X_mis,y_mis,__sensitive__,_logit_,X_critical,y_critical
    mis_predict = []
    X_mis = []
    y_mis = []
    __sensitive__ = []
    _logit_ = []
    X_critical = []
    y_critical = []

class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError


class Linear(Op):
    def __init__(self, input_size, output_size, name):
        self.params = {}
        self.name = name

        self.params['W'] = torch.randn((input_size, output_size), dtype=torch.float32)
        self.params['b'] = torch.zeros((1, output_size), dtype=torch.float32)

        self.inputs = None
        self.grads = {}

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        self.inputs = inputs
        return torch.matmul(inputs, self.params['W']) + self.params['b']

    def backward(self, grads):
        self.grads['W'] = torch.matmul(self.inputs.T, grads)
        self.grads['b'] = torch.sum(grads, dim=0)

        return torch.matmul(grads, self.params['W'].T)


class Logistic(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        outputs = 1.0 / (1.0 + torch.exp(-inputs))
        self.outputs = outputs
        return outputs

    def backward(self, grads):
        output_grad_inputs = torch.multiply(self.outputs, (1.0 - self.outputs))
        # print(output_grad_inputs.shape)
        return torch.multiply(grads, output_grad_inputs)


class Arctan(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        self.inputs = inputs
        outputs = torch.atan(inputs) * (2 / np.pi)
        self.outputs = outputs
        return self.outputs

    def backward(self, grads):
        self_grads = (2 / np.pi) * (1 / (1 + self.inputs ** 2))
        return torch.multiply(grads, self_grads)


class BinaryCrossEntropyLoss:
    def __init__(self, model):
        self.predicts = None
        self.labels = None
        self.num = None
        self.model = model

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]

        # self.predicts = torch.clamp(predicts, min=1e-7, max=1-1e-7)
        loss = -1. / self.num * (torch.matmul(self.labels.T, torch.log(self.predicts)) + torch.matmul(1 - self.labels.T,
                                                                                                      torch.log(
                                                                                                          1 - self.predicts)))
        loss = torch.squeeze(loss)
        return loss

    def backward(self):
        # print(f"labels shape:{self.labels.shape},predicts shape:{self.predicts.shape}")
        loss_grad_predicts = -1.0 * (self.labels / self.predicts - (1 - self.labels) / (1 - self.predicts)) / self.num
        # print(loss_grad_predicts.shape)
        self.model.backward(loss_grad_predicts)


'''

class Model_l2_characterized(Op):
    def __init__(self,input_size,hidden_size,output_size,**kwargs):
        self.layer1 = Linear(input_size, hidden_size, "l1")
        self.func1 = kwargs.get("func1",Arctan())
        self.layer2 = Linear(hidden_size, output_size, "l2")
        self.func2 = kwargs.get("func2",Logistic())

        self.layers = [self.layer1,self.func1,self.layer2,self.func2]

    def forward(self,inputs):
        z1 = self.layer1(inputs)
        a1 = self.func1(z1)
        z2 = self.layer2(a1)
        a2 = self.func2(z2)
        return a2


'''


class Model_L2(Op):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        self.layer1 = Linear(input_size, hidden_size, kwargs.get("l1_name", "l1"))
        self.func1 = Arctan()
        self.layer2 = Linear(hidden_size, output_size, kwargs.get("l1_name", "l2"))
        self.func2 = Logistic()
        # self.name = name

        self.layers = [self.layer1, self.func1, self.layer2, self.func2]

    def forward(self, inputs):
        z1 = self.layer1(inputs)
        a1 = self.func1(z1)
        z2 = self.layer2(a1)
        a2 = self.func2(z2)
        return a2

    def backward(self, loss_grad_a2):
        global loss_grad_character
        # print(loss_grad_a2.shape)
        b1 = self.func2.backward(loss_grad_a2)
        # print(b1.shape)
        b2 = self.layer2.backward(b1)
        b3 = self.func1.backward(b2)
        loss_grad_character = self.layer1.backward(b3)

    # def calculate_loss(self):

    def __show__(self):
        for layer in self.layers:
            if isinstance(layer.params, dict):
                print(f"{layer.name}'s params:{layer.params}")


class Model_L3_1(Op):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, **kwargs):
        self.layer1 = Linear(input_size, hidden_size_1, kwargs.get("l1_name", "l1"))
        self.func1 = Arctan()
        self.layer2 = Linear(hidden_size_1, hidden_size_2, kwargs.get("l2_name", "l2"))
        self.func2 = Arctan()
        self.layer3 = Linear(hidden_size_2, output_size, kwargs.get("l3_name", "l3"))
        self.func3 = Logistic()
        # self.name = name

        self.layers = [self.layer1, self.func1, self.layer2, self.func2, self.layer3, self.func3]

    def forward(self, inputs):
        z1 = self.layer1(inputs)
        a1 = self.func1(z1)
        z2 = self.layer2(a1)
        a2 = self.func2(z2)
        z3 = self.layer3(a2)
        a3 = self.func3(z3)
        return a3

    def backward(self, loss_grad_a3):
        global loss_grad_character
        # print(loss_grad_a2.shape)
        b1 = self.func3.backward(loss_grad_a3)
        # print(b1.shape)
        b2 = self.layer3.backward(b1)
        b3 = self.func2.backward(b2)
        # print(b1.shape)
        b4 = self.layer2.backward(b3)
        b5 = self.func1.backward(b4)
        loss_grad_character = self.layer1.backward(b5)


class BatchGD:
    def __init__(self, model, init_lr):
        self.model = model
        self.init_lr = init_lr

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class Runner:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn(model)

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_epochs = kwargs.get("log_epochs", 50)

        is_print = kwargs.get("is_print",True)
        is_save = kwargs.get("is_save",True)

        train_loss = []
        test_score = []
        test_score.append(0)

        # save_dir = kwargs.get("save_dir",r"D:\Project\python_code_\model")
        # save_dir = kwargs.get("save_dir", r"E:\DL_project\NN\model")
        save_dir = kwargs.get("save_dir", r"D:\git_storage\NN\model")

        for epoch in range(num_epochs):
            X, y = train_set
            logits = self.model(X)
            # print(f"logits shape:{logits.shape}")
            trn_loss = self.loss_fn(logits, y)
            train_loss.append(trn_loss)
            self.loss_fn.backward()
            # 反向传播
            self.optimizer.step()

            test_accuracy = self.evaluate(dev_set)

            '''if 0.9 <= test_accuracy and np.abs(test_accuracy - max(test_score)) <= 1e-5:
                print(f"{epoch} epoch:highest accuracy: {test_accuracy}")
                break'''
            if is_save:
                if test_accuracy >= 0.8 and test_accuracy >= max(test_score):
                    for layer in self.model.layers:
                        if isinstance(layer.params, dict):
                            torch.save(layer.params, os.path.join(save_dir, layer.name + "params.pth"))

            test_score.append(test_accuracy)
            if (epoch % log_epochs) == 0 and is_print:
                print(f"test accuracy:{test_accuracy:.4f}")

    # on "Thinkpad": model_dir = r"D:\git_storage\NN\model"
    def load_model(self, model_dir=r"D:\git_storage\NN\model"):
        # on Zhang: model_dir=r"E:\DL_project\NN\model"
        # def load_model(self, model_dir=r"E:\DL_project\NN\model"):
        # 获取所有层参数名称和保存路径之间的对应关系
        model_file_names = os.listdir(model_dir)
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pth", "")
            name_file_dict[name] = os.path.join(model_dir, file_name)

        # 加载每层参数
        for layer in self.model.layers:  # 遍历所有层
            if isinstance(layer.params, dict):
                name = layer.name + "params"
                file_path = name_file_dict[name]
                layer.params = torch.load(file_path)

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)  # 前向计算
        loss = self.loss_fn(logits, y)  # 计算损失
        # 计算预测类别
        preds = (logits >= 0.5).float()
        # 计算准确率
        accuracy = torch.mean((preds == y).float()).item()
        return accuracy

    def predict(self, data_loader,**kwargs):
        sum = 0
        total = 0
        is_print = kwargs.get("is_print",True)
        global mis_predict, X_mis, y_mis, X_critical, y_critical, _logit_
        for X_batch, y_batch in data_loader:
            for X_pred, y in zip(X_batch, y_batch):
                total += 1
                logits = self.model(X_pred)

                if 0.6 >= logits >= 0.4:
                    X_critical.append(X_pred)
                    y_critical.append(y)
                    _logit_.append(logits.item())

                preds = (logits >= 0.5).float()

                if preds.item() != y.item():
                    sum += 1
                    X_mis.append(X_pred)
                    y_mis.append(y)
                    mis_predict.append([X_pred, y])

                    if is_print:print(f"result is:{preds},wrong predict!")
                else:
                    if is_print:print(f"result is:{preds}")
        return sum/total

    def load_critical_data(self, **kwargs):
        global X_critical, y_critical, _logit_, __sensitive__
        X_critical_ = np.asarray(X_critical, dtype=np.float32)
        y_critical_ = np.asarray(y_critical, dtype=np.float32)
        X_batch = torch.tensor(X_critical_, dtype=torch.float32)
        y_batch = torch.tensor(y_critical_, dtype=torch.float32)
        dataset = TensorDataset(X_batch, y_batch)
        dataset = DataLoader(dataset, batch_size=1, shuffle=False)

        for X_batch_, y_batch_ in dataset:
            _sensitive_ = []
            X_batch_ = X_batch_.reshape([1, -1])
            y_batch_ = y_batch_.reshape([1, -1])

            logit = self.model(X_batch_)
            loss = self.loss_fn(logit, y_batch_)
            character_size = X_batch.shape[1]

            for i in range(character_size):
                X_batch_clone = X_batch_.clone()
                _epsilon_ = X_batch_clone[:, i] * 0.01
                _epsilon_ = torch.where(_epsilon_ == 0,
                                        torch.tensor(1e-6, dtype=torch.float32, device=_epsilon_.device), _epsilon_)
                X_batch_clone[:, i] += _epsilon_

                logit_2 = self.model(X_batch_clone)
                loss_2 = self.loss_fn(logit_2, y_batch_)

                sensitive = (loss_2.item() - loss.item()) / _epsilon_.item()
                # result = sensitive.tolist()
                result = np.round(sensitive, 4)

                _sensitive_.append(result)

            # print(_sensitive_)
            # print("\n")
            __sensitive__.append(_sensitive_)
        df1 = pd.DataFrame(__sensitive__).T
        df2 = pd.DataFrame(X_critical_).T
        df3 = pd.DataFrame(y_critical_).T
        df4 = pd.DataFrame(_logit_).T
        # save_path = r"D:\git_storage\NN\grad.xlsx"
        with pd.ExcelWriter("output(critical).xlsx") as writer:
            df1.to_excel(writer, sheet_name="sensitive", index=False)
            df2.to_excel(writer, sheet_name="X_critical", index=False)
            df3.to_excel(writer, sheet_name="y_critical", index=False)
            df4.to_excel(writer, sheet_name="logit", index=False)

    def show_sensitive(self, **kwargs):
        global y_mis, X_mis
        X_mis = np.asarray(X_mis, dtype=np.float32)  # 转换为 NumPy 数组
        y_mis = np.asarray(y_mis, dtype=np.float32)
        X_batch = torch.tensor(X_mis, dtype=torch.float32)
        y_batch = torch.tensor(y_mis, dtype=torch.float32)
        mis_dataset = TensorDataset(X_batch, y_batch)
        # _epsilon_ = X_batch * 0.01
        character_size = X_batch.shape[1]

        __sensitive__ = []
        for X_label, y_label in mis_dataset:
            _sensitive_ = []
            X_label = X_label.reshape([1, -1])
            y_label = y_label.reshape([-1, 1])

            logit = self.model(X_label)
            loss = self.loss_fn(logit, y_label)

            for i in range(character_size):
                X_label_clone = X_label.clone()
                _epsilon_ = X_label_clone[:, i] * 0.01
                X_label_clone[:, i] += _epsilon_

                logit_2 = self.model(X_label_clone)
                loss_2 = self.loss_fn(logit_2, y_label)

                sensitive = (loss_2.item() - loss.item()) / _epsilon_.item()
                # result = sensitive.tolist()
                result = np.round(sensitive, 4)

                _sensitive_.append(result)

            print(_sensitive_)
            print("\n")
            __sensitive__.append(_sensitive_)
        is_save = kwargs.get("is_save", 1)
        if is_save:
            df1 = pd.DataFrame(__sensitive__).T
            df2 = pd.DataFrame(X_mis).T
            df3 = pd.DataFrame(y_mis).T
            # save_path = r"D:\git_storage\NN\grad.xlsx"
            with pd.ExcelWriter("output.xlsx") as writer:
                df1.to_excel(writer, sheet_name="Sheet1", index=False)
                df2.to_excel(writer, sheet_name="Sheet2", index=False)
                df3.to_excel(writer, sheet_name="Sheet3", index=False)
            # df.to_excel(save_path,index = false)

    def show_data_sensitive(self, data):
        for X_batch, y_batch in data:
            print(f"\n")
            print(f"data gradient :")
            _sensitive_ = []

            X_batch = X_batch.reshape([1, -1])
            y_batch = y_batch.reshape([-1, 1])

            logit = self.model(X_batch)
            loss = self.loss_fn(logit, y_batch)
            character_size = X_batch.shape[1]

            for i in range(character_size):
                X_batch_clone = X_batch.clone()
                _epsilon_ = X_batch_clone[:, i] * 0.01
                _epsilon_ = torch.where(_epsilon_ == 0,
                                        torch.tensor(1e-6, dtype=torch.float32, device=_epsilon_.device), _epsilon_)
                X_batch_clone[:, i] += _epsilon_

                logit_2 = self.model(X_batch_clone)
                loss_2 = self.loss_fn(logit_2, y_batch)

                sensitive = (loss_2.item() - loss.item()) / _epsilon_.item()
                # result = sensitive.tolist()
                result = np.round(sensitive, 4)
                _sensitive_.append(result)
            print(_sensitive_)


def print_wrong_data():
    print(mis_predict)


def clear_wrong_data():
    global mis_predict
    mis_predict = []


def print_critical_data():
    global X_critical, y_critical
    X_critical_ = np.asarray(X_critical, np.float32)


# 加载breast_cancer数据集
breast_cancer = load_breast_cancer()
X = breast_cancer["data"]
y = breast_cancer["target"]

# y = y.reshape([-1,1])


# 对X中较大的数据进行处理
'''X[ :, X.max(axis=0) > 10] /= 10
X[ :, X.max(axis=0) > 10] /= 10
X[ :, X.max(axis=0) > 10] /= 5'''
# 将数据处理至相似数量级后，预测结果提升？

X_malignant = X[y == 0]
X_benign = X[y == 1]

mean_malignant = np.mean(X_malignant, axis=0)
mean_benign = np.mean(X_benign, axis=0)
std_malignant = np.std(X_malignant, axis=0)
std_benign = np.std(X_benign, axis=0)

y_malignant = np.array([1])
y_malignant = y_malignant.reshape([-1, 1])
y_benign = np.array([0])
y_benign = y_benign.reshape([-1, 1])

X_malignant_tensor = torch.tensor(mean_malignant, dtype=torch.float32)
X_benign_tensor = torch.tensor(mean_benign, dtype=torch.float32)
y_malignant_tensor = torch.tensor(y_malignant, dtype=torch.float32)
y_benign_tensor = torch.tensor(y_benign, dtype=torch.float32)
X_benign_tensor = X_benign_tensor.reshape([1, -1])
X_malignant_tensor = X_malignant_tensor.reshape([1, -1])

X_mean = torch.cat([X_malignant_tensor, X_benign_tensor], dim=0)
y_mean = torch.cat([y_malignant_tensor, y_benign_tensor], dim=0)
mean_dataset = TensorDataset(X_mean, y_mean)
mean_dataset = DataLoader(mean_dataset, batch_size=1, shuffle=False)
print(X_mean.shape)
print(y_mean.shape)

y = y.reshape([-1, 1])
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

'''
# 打印一个批次的数据
for batch in train_loader:
    X_batch, y_batch = batch
    print("X:", X_batch)
    print("y:", y_batch)
    break
'''

'''
#model = Model_L2(30,45,1,l1_name="l2_1",l2_name="l2_2")
#runner = Runner(model,BinaryCrossEntropyLoss,BatchGD(model,0.01))
#runner.train([X_train,y_train],[X_test,y_test],num_epochs=750,log_epochs=50)
#runner.load_model()

#runner.predict(test_loader)
#print_critical_data()

#model_2 = Model_L2(30,45,1)
#runner_2 = Runner(model_2,BinaryCrossEntropyLoss,BatchGD(model_2,0.01))
#runner_2.load_model()
#print(f"first_trained_model\n")
#runner_2.predict(test_loader)

#print_wrong_data()
#runner.show_sensitive()
#runner.load_critical_data()
'''

'''
print(f"first prediction:")
runner.predict(test_loader)
print(f"wrong data:")
print_wrong_data()
runner.loss_visualize()
'''

'''model_3_1 = Model_L2(30, 45, 1, l1_name="l3_1", l2_name="l3_2")
runner_3_1 = Runner(model_3_1, BinaryCrossEntropyLoss, BatchGD(model_3_1, 0.01))
runner_3_1.train([X_mean, y_mean], [X_test, y_test], num_epochs=100, log_epochs=10)
runner_3_1.show_data_sensitive(mean_dataset)
runner_3_1.train([X_train, y_train], [X_test, y_test], num_epochs=650, log_epochs=50)
runner_3_1.show_data_sensitive(mean_dataset)'''

first_model = []
second_model = []
third_model = []
forth_model = []

test_epoch = 100

def _standard_(params=3):
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)  # 0:良性, 1:恶性

    # 按类别分离数据
    X_benign = X[y == 0]  # 良性数据
    X_malignant = X[y == 1]  # 恶性数据

    # 计算均值和标准差
    mean_benign = X_benign.mean()
    std_benign = X_benign.std()

    mean_malignant = X_malignant.mean()
    std_malignant = X_malignant.std()

    # 计算 3 倍标准差范围
    lower_benign, upper_benign = mean_benign - params * std_benign, mean_benign + params * std_benign
    lower_malignant, upper_malignant = mean_malignant - params * std_malignant, mean_malignant + params * std_malignant

    # 过滤数据，仅保留在 3σ 范围内的数据
    filtered_benign = X_benign[(X_benign >= lower_benign) & (X_benign <= upper_benign)].dropna()
    filtered_malignant = X_malignant[(X_malignant >= lower_malignant) & (X_malignant <= upper_malignant)].dropna()

    # 输出筛选后的数据大小
    #print(f"原始良性样本数: {X_benign.shape[0]}, 过滤后: {filtered_benign.shape[0]}")
    #print(f"原始恶性样本数: {X_malignant.shape[0]}, 过滤后: {filtered_malignant.shape[0]}")
    #重新封装
    X_filtered = pd.concat([filtered_benign, filtered_malignant], ignore_index=True)
    y_filtered = pd.Series([0] * len(filtered_benign) + [1] * len(filtered_malignant))
    X_filtered = X_filtered.to_numpy()
    y_filtered = y_filtered.to_numpy()
    y_filtered = y_filtered.reshape([-1, 1])

    X_filtered = torch.tensor(X_filtered, dtype=torch.float32)
    y_filtered = torch.tensor(y_filtered, dtype=torch.float32)

    return [X_filtered,y_filtered]

    # 可选：导出 CSV
    # filtered_benign.to_csv("filtered_benign.csv", index=False)
    # filtered_malignant.to_csv("filtered_malignant.csv", index=False)


X_filtered_,y_filtered_ = _standard_()
X_filtered_train,X_filtered_test,y_filtered_train,y_filtered_test = train_test_split(X_filtered_,y_filtered_,test_size=0.2,random_state=42)
filtered_dataset = TensorDataset(X_filtered_test,y_filtered_test)
filtered_test_loader = DataLoader(filtered_dataset,batch_size=16,shuffle=False)

for i in range(test_epoch):

    model = Model_L3_1(30,45,10,1)
    runner = Runner(model,BinaryCrossEntropyLoss,BatchGD(model,0.01))
    runner.train([X_mean, y_mean], [X_test, y_test], num_epochs=20, log_epochs=5,is_print=False,is_save=False)
    runner.train([X_train, y_train], [X_test, y_test], num_epochs=1000, log_epochs=100,is_print=False,is_save=False)
    predict = runner.predict(test_loader,is_print=False)
    first_model.append(predict)

    reset()

    model_2 = Model_L3_1(30,45,10,1)
    runner_2 = Runner(model_2, BinaryCrossEntropyLoss, BatchGD(model_2, 0.01))
    runner_2.train([X_train, y_train], [X_test, y_test], num_epochs=500, log_epochs=100,is_print=False,is_save=False)
    predict_2 = runner_2.predict(test_loader,is_print=False)
    second_model.append(predict_2)

    reset()

    model_3 = Model_L3_1(30, 45, 10, 1)
    runner_3 = Runner(model_3, BinaryCrossEntropyLoss, BatchGD(model_3, 0.01))
    runner_3.train([X_mean, y_mean], [X_test, y_test], num_epochs=20, log_epochs=5, is_print=False, is_save=False)
    runner_3.train([X_filtered_train,y_filtered_train], [X_filtered_test,y_filtered_test], num_epochs=1000, log_epochs=100, is_print=False, is_save=False)
    predict_3 = runner_3.predict(test_loader, is_print=False)
    third_model.append(predict_3)

    reset()

    model_4 = Model_L3_1(30, 45, 10, 1)
    runner_4 = Runner(model_4, BinaryCrossEntropyLoss, BatchGD(model_4, 0.01))
    runner_4.train([X_mean, y_mean], [X_test, y_test], num_epochs=20, log_epochs=5, is_print=False, is_save=False)
    runner_4.train([X_filtered_train, y_filtered_train], [X_filtered_test, y_filtered_test], num_epochs=1000,
                   log_epochs=100, is_print=False, is_save=False)
    predict_4 = runner_4.predict(filtered_test_loader, is_print=False)
    forth_model.append(predict_4)

    reset()

    print(f"epoch{i+1} finished.")

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# 创建 2 行 2 列的子图
fig, axes = plt.subplots(2, 2, figsize=(10,6))  # 这里 axes 是一个 2x2 的 numpy 数组

# 绘制第一个模型的直方图
sns.histplot(first_model, bins=30, alpha=0.4, label="model one", kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Model One")

# 绘制第二个模型的直方图
sns.histplot(second_model, bins=30, alpha=0.4, label="model two", kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Model Two")

# 绘制第三个模型的直方图
sns.histplot(third_model, bins=30, alpha=0.4, label="model three", kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Model Three")

# 绘制第四个模型的直方图
sns.histplot(forth_model, bins=30, alpha=0.4, label="model four", kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Model Four")

plt.title("4个数据分布 - KDE 密度估计")
plt.xlabel("数值")
plt.ylabel("密度")
plt.legend()

plt.tight_layout()
plt.show()

