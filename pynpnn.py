import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, w, b): 
        self.w = w
        self.b = b
    def feedforward(self, inputs):
        a = np.dot(self.w, inputs) + self.b
        return sigmoid(a)
    
w = np.array([0, 1])
b = 4
n = Neuron(w, b)

x = np.array([2, 3])
print(n.feedforward(x))

class NeuralNetwork:
    def __init__(self):
        w = np.array([0, 1])
        b = 0
        self.h1 = Neuron(w, b)
        self.h2 = Neuron(w, b)
        self.o = Neuron(w, b)
        
    def feedforward(self, x):
        o_h1 = self.h1.feedforward(x)
        o_h2 = self.h2.feedforward(x)
        o_o = self.o.feedforward(np.array([o_h1, o_h2]))
        return o_o
    
net = NeuralNetwork()
x = np.array([2, 3])
net.feedforward(x)
# np.exp(-1)
# import matplotlib.pyplot as plt
# x = np.linspace(-2, 2, 100)
# y = np.exp(-x)
# plt.plot(x, y)

def mse_loss(y_t, y_p):
    return ((y_t - y_p) ** 2).mean()
def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)
class NeuralNet:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        
    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o
    
    def train(self, data, y_true):
        lr = 0.1
        epochs = 1000
        for epoch in range(epochs):
            for x, y_t in zip(data, y_true):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_p = o1
                
                d_L_yp = -2 * (y_t - y_p)
                
                d_yp_w5 = h1 * d_sigmoid(sum_o1)
                d_yp_w6 = h2 * d_sigmoid(sum_o1)
                d_yp_b3 = d_sigmoid(sum_o1)
                d_yp_h1 = self.w5 * d_sigmoid(sum_o1)
                d_yp_h2 = self.w6 * d_sigmoid(sum_o1)
                
                d_h1_w1 = x[0] * d_sigmoid(sum_h1)
                d_h1_w2 = x[1] * d_sigmoid(sum_h1)
                d_h1_b1 = d_sigmoid(sum_h1)
                
                d_h2_w3 = x[0] * d_sigmoid(sum_h2)
                d_h2_w4 = x[0] * d_sigmoid(sum_h2)
                d_h2_b2 = d_sigmoid(sum_h2)
                
                #update
                self.w1 -= lr * d_L_yp * d_yp_h1 * d_h1_w1
                self.w2 -= lr * d_L_yp * d_yp_h1 * d_h1_w2
                self.b1 -= lr * d_L_yp * d_yp_h1 * d_h1_b1
                
                self.w3 -= lr * d_L_yp * d_yp_h2 * d_h2_w3
                self.w4 -= lr * d_L_yp * d_yp_h2 * d_h2_w4
                self.b2 -= lr * d_L_yp * d_yp_h2 * d_h2_b2
                
                self.w5 -= lr * d_L_yp * d_yp_w5
                self.w6 -= lr * d_L_yp * d_yp_w6
                self.b3 -= lr * d_L_yp * d_yp_b3
                
                if epoch % 100 == 0:
                    y_ps = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(y_true, y_ps)
                    print("Epoch %d Loss: %.3f" % (epoch, loss))
                    
                # Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])
nn = NeuralNet()
nn.train(data, y_trues)                 
# y_t = np.array([1, 0, 0, 1])
# y_p = np.array([0, 0, 0, 0])
# mse_loss(y_t, y_p)


emy = np.array([-7, -3])
frk = np.array([20, 2])
nn.feedforward(emy), nn.feedforward(frk)