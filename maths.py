import math

def sigmoid(x):
    x = max(-700, min(700, x))
    return 1/(1+math.exp(-x))

def derived_sigmoid(output):
    return output*(1-output)