import numpy as np

def get_score(prob, alpha=6.0):
    mean_prob = 0.59
    std_prob = 0.32
    z = (prob - mean_prob) / std_prob
    sigmoid = 1 / (1 + np.exp(-z))
    a_score = alpha * sigmoid
    return round(a_score, 2)


#ver1
#mean -> 0.5888821500745341
#std -> 0.2377276730579365

#ver2
# 0.5893650227254659
# 0.3228613814047024
