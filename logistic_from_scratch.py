import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class_1 = np.random.random(20)*2 + 1
class_2 = np.random.random(20)*2 - 0.5
data = pd.DataFrame()
data['x'] = np.concatenate([class_1, class_2])
data['y'] = [0]*20 + [1]*20
data = data.sample(frac=1)
data.head()

plt.scatter(data.x, data.y, s=5)
plt.show()

def calculate_gradient_log_likelihood(curr_betas, data):
    numerator = np.exp(curr_betas[0] + curr_betas[1]*data.x)
    p = numerator / (1 + numerator)
    
    partial_0 = np.sum(data.y - p)
    partial_1 = np.sum((data.y - p)*data.x)
    
    return np.array([partial_0, partial_1])
