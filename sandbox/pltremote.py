import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.array([1,2,3]))
#plt.show()
plt.savefig('test.png')
