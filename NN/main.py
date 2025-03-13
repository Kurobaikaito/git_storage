import torch
import os
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

mis_predict = []
loss_grad_character = None



breast_cancer = load_breast_cancer()
X = breast_cancer["data"]
y = breast_cancer["target"]
y = y.reshape([-1,1])

X_tensor = torch.tensor(X,dtype=torch.float32)
y_tensor = torch.tensor(y,dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X_tensor,y_tensor,test_size=0.2,random_state=42)

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

class Op(object):
    def __init__(self):
        pass

    def __call__(self,inputs):
        return self.forward(inputs)

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,inputs):
        raise NotImplementedError

class Linear(Op):
    def __init__(self,input_size,output_size,name):

        self.params = {}
        self.name = name

        self.params['W'] = torch.randn((input_size,output_size),dtype=torch.float32)
        self.params['b'] = torch.zeros((1,output_size),dtype=torch.float32)

        self.inputs = None
        self.grads = {}


    def __call__(self,inputs):
        return self.forward(inputs)

    def forward(self,inputs):
        self.inputs = inputs
        return torch.matmul(inputs,self.params['W']) + self.params['b']

    def backward(self,grads):
        self.grads['W'] = torch.matmul(self.inputs.T,grads)
        self.grads['b'] = torch.sum(grads,dim=0)

        return torch.matmul(grads,self.params['W'].T)

class Logistic(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self,inputs):
        outputs = 1.0 / (1.0 + torch.exp(-inputs))
        self.outputs = outputs
        return outputs

    def backward(self,grads):
        output_grad_inputs = torch.multiply(self.outputs,(1.0 - self.outputs))
        #print(output_grad_inputs.shape)
        return torch.multiply(grads,output_grad_inputs)

class Arctan(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def __call__(self,inputs):
        return self.forward(inputs)

    def forward(self,inputs):
        self.inputs = inputs
        outputs = torch.atan(inputs)*(2/np.pi)
        self.outputs = outputs
        return self.outputs

    def backward(self,grads):
        self_grads = (2/np.pi)*(1/(1+self.inputs**2))
        return torch.multiply(grads,self_grads)

class BinaryCrossEntropyLoss:
    def __init__(self,model):
        self.predicts = None
        self.labels = None
        self.num = None
        self.model = model

    def __call__(self,predicts,labels):
        return self.forward(predicts,labels)

    def forward(self,predicts,labels):
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]

        #self.predicts = torch.clamp(predicts, min=1e-7, max=1-1e-7)
        loss = -1. / self.num*(torch.matmul(self.labels.T,torch.log(self.predicts)) + torch.matmul(1-self.labels.T,torch.log(1-self.predicts)))
        loss = torch.squeeze(loss)
        return loss

    def backward(self):
        #print(f"labels shape:{self.labels.shape},predicts shape:{self.predicts.shape}")
        loss_grad_predicts = -1.0 * (self.labels / self.predicts - (1 - self.labels) / (1 - self.predicts)) / self.num
        #print(loss_grad_predicts.shape)
        self.model.backward(loss_grad_predicts)

class Model_L2(Op):
    def __init__(self,input_size,hidden_size,output_size,name):
        self.layer1 = Linear(input_size,hidden_size,"l1")
        self.func1 = Arctan()
        self.layer2 = Linear(hidden_size,output_size,"l2")
        self.func2 = Logistic()
        self.name = name

        self.layers = [self.layer1,self.func1,self.layer2,self.func2]

    def forward(self,inputs):
        z1 = self.layer1(inputs)
        a1 = self.func1(z1)
        z2 = self.layer2(a1)
        a2 = self.func2(z2)
        return a2

    def backward(self,loss_grad_a2):
        global loss_grad_character
        #print(loss_grad_a2.shape)
        b1 = self.func2.backward(loss_grad_a2)
        #print(b1.shape)
        b2 = self.layer2.backward(b1)
        b3 = self.func1.backward(b2)
        loss_grad_character = self.layer1.backward(b3)

    def __show__(self):
        for layer in self.layers:
            if isinstance(layer.params,dict):
                print(f"{layer.name}'s params:{layer.params}")


class BatchGD:
    def __init__(self,model,init_lr):
        self.model = model
        self.init_lr = init_lr

    def step(self):
        for layer in self.model.layers:
            if isinstance(layer.params,dict):
                for key in layer.params.keys():
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class Runner:
    def __init__(self,model,loss_fn,optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn(model)

    def train(self,train_set,dev_set,**kwargs):
        num_epochs = kwargs.get("num_epochs",0)
        log_epochs = kwargs.get("log_epochs",50)

        train_loss = []
        test_score = []

        #save_dir = kwargs.get("save_dir",r"D:\Project\python_code_\model")
        save_dir = kwargs.get("save_dir", r"E:\DL_project\NN\model")

        for epoch in range(num_epochs):
            X,y = train_set
            logits = self.model(X)
            #print(f"logits shape:{logits.shape}")
            trn_loss = self.loss_fn(logits,y)
            train_loss.append(trn_loss)
            self.loss_fn.backward()
            #反向传播
            self.optimizer.step()

            test_accuracy = self.evaluate(dev_set)
            if test_accuracy >= 0.8 and test_accuracy >= max(test_score):
                for layer in self.model.layers:
                    if isinstance(layer.params,dict):
                        torch.save(layer.params,os.path.join(save_dir,layer.name+"params.pth"))

            test_score.append(test_accuracy)
            if (epoch%log_epochs) == 0:
                print(f"test accuracy:{test_accuracy:.4f}")

    def load_model(self, model_dir=r"E:\DL_project\NN\model"):
        # 获取所有层参数名称和保存路径之间的对应关系
        model_file_names = os.listdir(model_dir)
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pth","")
            name_file_dict[name] = os.path.join(model_dir, file_name)

        # 加载每层参数
        for layer in self.model.layers: # 遍历所有层
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

    def predict(self,data_loader):
        for X_batch,y_batch in data_loader:
            for X_pred,y in zip(X_batch,y_batch):
                logits = self.model(X_pred)
                preds = (logits >= 0.5).float()

                if preds.item() != y.item():
                    mis_predict.append([X_pred,y])
                    print(f"result is:{preds},wrong predict!")
                else:
                    print(f"result is:{preds}")

    def loss_visualize(self):
        global mis_predict,loss_grad_character
        for X_,y_ in mis_predict:
            print(f"\n")
            X_batch = X_.reshape([1,-1])
            y_batch = y_.reshape([-1,1])
            #print(X_batch.shape,y_batch.shape)

            logits = self.model(X_batch)
            loss = self.loss_fn(logits,y_batch)
            self.loss_fn.backward()

            print(f"label:{y_batch}")
            print(f"grad:{loss_grad_character}")
            print(f"data:{X_batch}")

            #loss_per_character_ = torch.tensor(loss_grad_character,dtype=torch.float32)
            #X_t = torch.tensor(X_batch,dtype=torch.float32)
            print(f"loss in each character:{loss_grad_character * X_batch}\n")


def print_wrong_data():
    print(mis_predict)

def clear_wrong_data():
    global mis_predict
    mis_predict = []



model = Model_L2(30,45,1,"test")
runner = Runner(model,BinaryCrossEntropyLoss,BatchGD(model,0.01))
#runner.train([X_train,y_train],[X_test,y_test],num_epochs=500,log_epochs=50)
runner.load_model()
#model.__show__()
print(f"first prediction:")
runner.predict(test_loader)
print(f"wrong data:")
print_wrong_data()
runner.loss_visualize()
