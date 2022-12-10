import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, np.pi, 1000)
y = np.sin(x)

plt.subplot(221).plot(x, y)
plt.subplot(223).plot(x, y)
plt.subplot(122).plot(x, -y)

plt.show()
