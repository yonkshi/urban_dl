import numpy as np
import matplotlib.pyplot as plt
from eolearn.core import EOTask, EOPatch

patch = EOPatch.load('data/slovenia/eopatch-14x8/')

dat = patch.data['BANDS']

img = dat[50, ..., :3]
plt.imshow(img)
plt.title('RGB')
plt.show()

img = dat[0, ..., :3]
plt.imshow(img)
plt.title('IR (MIR')
plt.show()

print('hello')