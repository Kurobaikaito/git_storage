import paddle
import matplotlib.pyplot as plt

#logistic function and Tanh function
def logistic(z):
    return 1.0 / (1.0 + paddle.exp(-z))

def Tanh(z):
    return (paddle.exp(z) - paddle.exp(-z)) / (paddle.exp(z) + paddle.exp(-z))

z - paddle.linspace(-10,10,10000)
plt.figure()
plt.plot(z.tolist(),logistic(z).tolist(),color = 'red',label = "logistic function")
plt.plot(z.tolist(),Tanh(z).tolist(),color = 'b',linestyle = '--',label = "Tanh function")
ax.plt.gca()

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))
plt.legend()
plt.show()