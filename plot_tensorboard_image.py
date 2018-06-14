import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('results/final_1/run_.-tag-loss_1.csv')

loss = np.array(data['Value'])
iteration = np.array(data['Step'])

plt.figure()
plt.semilogy(iteration,loss)

data2 = pd.read_csv('results/final_3/run_.-tag-loss_1.csv')

loss2 = np.array(data2['Value'])
iteration2 = np.array(data2['Step'])

plt.semilogy(iteration2,loss2)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend(['windows 5-1','windows 2-2'])
plt.show()