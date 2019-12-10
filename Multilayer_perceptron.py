import numpy as np


#NoNeorons = 2
#NoFeatures = 2


def initialize_weights(no_hidden_layers, no_input,  no_neurons_each_layer, no_output):
    weights = dict()
    bias = dict()
    weights[1] = np.random.rand(no_neurons_each_layer[0], no_input)
    for i in range(1, no_hidden_layers):
        weights[i + 1] = np.random.rand(no_neurons_each_layer[i], no_neurons_each_layer[i - 1])
    weights[no_hidden_layers + 1] = np.random.rand(no_output, no_neurons_each_layer[no_hidden_layers - 1])

    for i in range(no_hidden_layers):
        bias[i + 1] = np.random.rand(no_neurons_each_layer[i])
    bias[no_hidden_layers + 1] = np.random.rand(no_output)
    return weights, bias


def sigmoid_derivative(net_vals):
    res = []
    for item in net_vals:
        res.append(item * (1 - item))
    return res


def forward(input, weights, bias_weights, biasval, activation_func):
    nodes = dict()
    for i in weights:
        if i == 1:
            dot = np.dot(weights[i], input)
            dot += (biasval * bias_weights[i])
        else:
            dot = np.dot(weights[i], nodes[i - 1])
            dot += (biasval * bias_weights[i])
        nodes[i] = select_and_apply_activation(dot, activation_func)
    return nodes


def sigmoid(net_vals):
    arr = []
    for e in net_vals:
        temp = 1/(1+np.exp(-e))
        arr.append(temp)
#    print(np.shape(arr))
    return arr


def select_and_apply_activation_derivative(net_vals, activ_fun):
    if activ_fun == 0:
        arr = sigmoid_derivative(net_vals)
    else:
        arr = tanh_derivative(net_vals)
    return arr


def tanh(net_vals):
    arr = []
    for element in net_vals:
        temp = np.divide((1 - np.exp(-element)), (1+np.exp(-element)))
        arr.append(temp)

    return arr


def tanh_derivative(net_vals):
    val = np.subtract(1, np.power(tanh(net_vals), 2))
    return val


def select_and_apply_activation(net_vals, function):
    if function == 0:
        arr = sigmoid(net_vals)
    else:
        arr = tanh(net_vals)
    return arr


def backward(desired, nodes, weights, activ_func):
    error = dict()
    error[len(nodes)] = np.multiply(np.subtract(desired, nodes[len(nodes)]), select_and_apply_activation_derivative(nodes[len(nodes)], activ_func))
    for i in reversed(range(len(nodes) - 1)):
        error[i + 1] = np.multiply(np.dot(error[i + 2], weights[i + 2]), select_and_apply_activation_derivative(nodes[i + 1], activ_func))

    return error


def update_weights(weights, learning_rate, error, input, neurons_per_layer, nodes, NoOutput, bias, bias_val):
    for i in weights:
        arr = []
        if i == 1:
            for j in range(neurons_per_layer[i - 1]):
               # print(np.shape(error[1]))
              #  print(input)
                temp = np.multiply(np.multiply(learning_rate, error[1][j]), input)
                arr.append(temp)
            weights[i] = np.add(weights[1], arr)
        elif i == len(weights):
            for j in range(NoOutput):
                temp = np.multiply(np.multiply(learning_rate, error[i][j]), nodes[i - 1])
                arr.append(temp)
            weights[i] = np.add(weights[i], arr)
        else:
            for j in range(neurons_per_layer[i - 1]):
                temp = np.multiply(np.multiply(learning_rate, error[i][j]), nodes[i - 1])
                arr.append(temp)
            weights[i] = np.add(weights[i], arr)
    for i in bias:  # updating bias
        bias[i] = bias[i] + np.multiply(np.multiply(bias_val, learning_rate), error[i])
    return weights, bias


def read_dataset():
    f = open('C:\\Users\\User\\Desktop\\neuralTask1\\IrisData.txt', 'r')
    f.readline()
    D = np.zeros((150, 4))
    lines = f.readlines()
    counter = 0
    temp = np.zeros((1, 5))
    for line in lines:
        temp = line.split(",")
        D[counter, 0:4] = temp[0:4]
        counter += 1
    return D


def split_shuffle_data(whole_data):
    train_data = np.zeros((90, 4))
    test_data = np.zeros((60, 4))
    train_labels = np.zeros((90, 3))
    test_labels = np.zeros((60, 3))

    train_data[0:30] = whole_data[0:30]
    train_data[30:60] = whole_data[50:80]
    train_data[60:90] = whole_data[100:130]

    train_labels[0:30] = [1, 0, 0]
    train_labels[30:60] = [0, 1, 0]
    train_labels[60:90] = [0, 0, 1]

    test_data[0:20] = whole_data[30: 50]
    test_data[20: 40] = whole_data[80:100]
    test_data[40:60] = whole_data[130:150]

    test_labels[0:20] = [1, 0, 0]
    test_labels[20:40] = [0, 1, 0]
    test_labels[20:40] = [0, 1, 0]
    test_labels[40:60] = [0, 0, 1]

    return train_data, train_labels, test_data, test_labels


def train(train_data, num_input, num_output, num_hidden_layers, desired_output, neurons_per_layer, lr, bias_val, num_epochs, activ_function):
    weights, bias_weights = initialize_weights(num_hidden_layers, num_input, neurons_per_layer, num_output)

    for j in range(num_epochs):
        for i in range(90):
            net_vals = forward(train_data[i], weights, bias_weights, bias_val, activ_function)
            error = backward(desired_output[i], net_vals, weights, activ_function)
            weights, bias_weights = update_weights(weights, lr, error, train_data[i], neurons_per_layer, net_vals, num_output, bias_weights, bias_val)  #updated_weights
            for x in weights:
                print(weights[x])
            print("---------------------------------")
    return weights, bias_weights


def set_class(net_vals):
    for i in range(60):
        index = np.argmax(net_vals[i])
        net_vals[i] = np.zeros(3)
        net_vals[i][index] = 1
    return net_vals


def test(test_data, test_labels, final_weights, final_bias_weights, bias_val, NoHiddenLayers, sigmoid_function):
    output = []
    for i in range(60):
        net_vals = forward(test_data[i], final_weights, final_bias_weights, bias_val, sigmoid_function)
        output.append(net_vals[NoHiddenLayers + 1])
    final_output = set_class(output)
    return final_output


def main( num_hidden_layers,learning_rate,bias_val,activation_func,num_epochs):
    num_input = 4  # number of inout neurons always 4
    num_output = 3  # number of output neurons always 3
    num_hidden_layers = 2 # number of hidden layers from GUI
    neurons_per_layer = [2, 3]  # number of layers per layer without input and output len(NeuronsPerLayer)= NoHiddenLayers
    #    BiasWeights = dict()
    learning_rate = 0.01  # learning rate from GUI
    bias_val = 1 #  bias value from GUI 0 or 1
    activation_func = 1
    num_epochs = 100   #  number of epochs from GUI
    whole_data = read_dataset()
    train_data, train_labels, test_data, test_labels = split_shuffle_data(whole_data)

    final_weights, final_bias_weights = train(train_data, num_input, num_output, num_hidden_layers, train_labels, neurons_per_layer, learning_rate, bias_val, num_epochs, activation_func)

    print("final weights")
    print("================================================================")
    for e in final_weights:
        print(e)
    print("=================================================================")
    net_values = test(test_data, test_labels, final_weights, final_bias_weights, bias_val, num_hidden_layers, activation_func)



 #   print("=================================================")
    for e in final_weights:
        print(e)
        print(final_weights[e])


#main()


"""
#3amal classify shwaya sa7 
NoInput = 4
NoOutput = 3
NoHiddenLayers = 1
NeuronsPerLayer = [2]
BiasWeights = dict()
Lr = 0.00000000000000000001
BiasVal = 1
"""



"""
Weights = dict()
Nodes = dict()
BiasWeights = dict()
updated_weights = dict()


Weights[1] = [[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6]]
Weights[2] = [[0.7, 0.8, 2],
              [0.9, 1.0, 2.1]]
Weights[3] = [[1.1, 1.2],
              [1.3, 1.4]]

Nodes[1] = [3, 4, 5]
Nodes[2] = [6, 7]
Nodes[3] = [8, 9]

BiasWeights[1] = [0.2, 0.4, 0.6]
BiasWeights[2] = [0.8, 1.0]
BiasWeights[3] = [1.2, 1.4]

print("updated weights")
    for w in updated_weights:
        print(w)
        print(updated_weights[w])

    print("Weights : ")
    for e in weights:
        print(e)
        print(weights[e])
        print()
    print("=============================================================")
    print("bias")
    for e in bias_weights:
        print(e)
        print(bias_weights[e])
        print()
    print("=============================================================")

    print("nodes")
    for e in net_vals:
        print(e)
        print(net_vals[e])
        print()
    print("=============================================================")

    print("error")
    for e in error:
        print(e)
        print(error[e])
        print()
print("=============================================================")


"""
