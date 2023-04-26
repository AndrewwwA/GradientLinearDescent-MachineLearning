import copy, math
import numpy as np

# CAN CHANGE TRAINING DATA HERE <><><><><><><><><><><><><><><><><><><>
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# CAN CHANGE INITAL VALEUS FOR W AND B HERE <><><><><><><><><><><><><><><>
initial_w = np.array([1, 10, -50, -20])
initial_b = 500

# LEARNING RATE HERE <><><><><><><><><><><><><><><>
alpha = 8.0e-7
iterations = 50000

def computeGradient(x, y, w, b):
    m,n = x.shape
    #GRABS THE PARAMETER VALUES NOT ROWS WITH N
    gradW = np.zeros(n)
    gradB = 0
    # LOOPS THROUGH EACH ROW
    for i in range(m):
        diff = (np.dot(x[i], w) + b) - y[i]
        # LOOPS THROUGH EACH COLUMN
        for j in range(n):
            # Calculates gradient of Cost for each spot in the column and row
            gradW[j] = gradW[j] + diff * x[i, j]     
        # Calcs B graident of Cost
        gradB = gradB + diff
    # Divides by M (number of examples)
    gradW = gradW / m
    gradB = gradB / m
    
    return gradW, gradB

def gradientDesc(x, y, wInt, bInt, alpha, numTimes):
    
    # COPIES THE VALUES WITHOUT CHANGING IT
    w = copy.deepcopy(wInt)
    b = bInt
    
    # CHANGES OVER 10k ITERATIONS
    for i in range(numTimes):
        gradW, gradB = computeGradient(x, y, w, b)
        
        w = w - alpha * gradW
        b = b - alpha * gradB
        
        # if i% math.ceil(numTimes / 10) == 0:
        #     print(f"Iteration {i}")

    return w, b
    
wFin, bFin = gradientDesc(x_train, y_train, initial_w, initial_b, alpha, iterations)
size = x_train.shape
for i in range(size[0]):
    print(f"Guess: {np.dot(x_train[i], wFin) + bFin} actualVal: {y_train[i]}")