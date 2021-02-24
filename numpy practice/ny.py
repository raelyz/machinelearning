import numpy as np

# Your goal is to create a simple dataset consisting of a single feature and a label as follows:
# Assign a sequence of integers from 6 to 20 (inclusive) to a NumPy array named feature.
# Assign 15 values to a NumPy array named label such that:
# label = (3)(feature) + 4
# For example, the first value for label should be:
# label = (3)(6) + 4 = 22

feature = np.arange(6, 21)
label = np.array(3*feature + 4)
print(feature)
print(label)
noise = 2*(np.random.random([15])-np.random.random([15]))
print(noise)
label = label + noise
print(label)
