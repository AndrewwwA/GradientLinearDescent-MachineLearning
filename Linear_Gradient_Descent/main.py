import math
import numpy as np
import matplotlib.pyplot as plt

# Data to analyze and make a model out of

x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([30000.0, 60000.0, 85000.0, 130000.0])

# J(w, b) = ( 𝐽(𝑤,𝑏)=12𝑚∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2 )
def calculateCost(x, y, w, b):
    
    # Tells how many times to run the cost function (training data)
    trainingDataAmount = x.shape[0]
    # Stores the cost overall
    cost = 0
    
    for i in range(trainingDataAmount):
        preCostWB = w * x[i] + b
        # Calculates the cost (pretty much difference from the predicition and actual value)
        cost = cost + (preCostWB - y[i])**2
    # Calculates the cost of all training data (SIGMA) and then divided it by the amount for a reason I'm not sure but it makes the math work
    totalCost = 1 / (2 * trainingDataAmount) * cost
    return totalCost

# Calculate the gradient (x and y are the training data., w and b are the weights and bias) allows you to find the values of w and b
def calculateGradient(x, y, w, b):
    
    # Tells how many times to run the cost function (training data)
    trainingDataAmount = x.shape[0]
    # stores change amounts (gradients) for all cases of w and b
    graW = 0
    graB = 0
    
    for i in range(trainingDataAmount):
        # gets cost
        preCostWB = w * x[i] + b
        # x[i] becuase derivitaives cancel out the 2x[i] initially 
        singGradW = (preCostWB - y[i]) * x[i]
        # Doesn't include x[i] because MATH
        singGradB = preCostWB - y[i]
        # Adds both values to total overall
        graW += singGradW
        graB += singGradB
    # Finally you divided by 2m but cancels out becuase of the derivitaive 2x[i] as seen on line 38
    graW =graW / trainingDataAmount
    graB =graB / trainingDataAmount
    # Finally return both total values
    return graW, graB

# alpha - learning rate intVal = starting value so if first time it's 0.
def gradientDescent(x, y, intValW, intValB, alpha, numIterations):
    #Performs gradient descent to change w and b by completing the calculation numInterations times using learning rate alpha
    
    #Storing cash history +  parameter history
    cHistory = []
    pHistory = []
    newB = intValB
    newW = intValW
    
    # looping through ? number of interations
    for i in range(numIterations):
        # gets the prediction 
        graW, graB = calculateGradient(x, y, newW, newB)
        
        # changes the values to the next change
        newB = newB - alpha * graB
        newW = newW - alpha * graW
        
        # adds everything to history
        cHistory.append(calculateCost(x, y, newW, newB))
        pHistory.append([newW, newB])

    return newW, newB, cHistory, pHistory


wStart = 0
bStart = 0
iterations = 10000
alphaVal = 1.0e-2

finalW, finalB, cHistory, pHistory = gradientDescent(x_train, y_train, wStart, bStart, alphaVal, iterations)
# Change what number to check here (3.0 at the start)
finalAnswer = math.ceil(finalW * 3.0 + finalB)
print(finalAnswer)
        
        # Meh looking graph 
plt.scatter(x_train, y_train, color='blue', label='Training data')
x_pred = np.linspace(0, 3, 100) 
y_pred = finalW * x_pred + finalB 
plt.plot(x_pred, y_pred, color='red', label='Model prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Model')
plt.legend()

plt.show()
    
    
# old testing stuff while making the model

# cost = calculateCost(x_train, y_train, 0, 0)
# graW, graB = calculateGradient(x_train, y_train, 0, 0)
# print(graW, graB)
# plt.plot(graW, graB)
# plt.xlabel('m')
# plt.ylabel('c')
# plt.show()

# calculateCost(x_train, y_train, 0, 0)