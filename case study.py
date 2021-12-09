import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel('cum_case.xlsx')
value = df['newCasesByPublishDate'].values[160:1000]
print(len(value))
x = []
y = []
seq = 3
for i in range(len(value)-seq-1):
    x.append(value[i:i+seq])
    y.append(value[i+seq])
plt.plot(value[3:], label='True Value')
plt.title('new cases (England)')
plt.xlabel('Day')
plt.ylabel('New Cases')
plt.show()