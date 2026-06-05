import numpy as np

data = np.load("data/run_model_results.npz")


Z = data["Z"]
predictions = data["predictions"]

mae = np.mean(np.abs(Z[1:] - predictions[:-1]))
mse = np.mean((Z[1:] - predictions[:-1])**2)


print(f'{mae=}')
print(f'{mse=}')

import matplotlib.pyplot as plt

plt.plot(Z[:,6], label="True Data")
plt.plot(predictions[:,6], label="Prediction")
plt.legend()
plt.show()