import numpy as np
from optimal_offline import KServer

s = np.random.normal(0, 1, (10, 3))
r = np.random.normal(0, 1, (10000, 3))

instance = KServer(s, r, 2)
print(instance.optimal_cost())