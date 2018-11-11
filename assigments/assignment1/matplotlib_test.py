import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()

imagepath = 'q3_word_vectors.png'
im1 = Image.open(imagepath)
im1 = np.array(im1)#获得numpy对象,RGB
plt.imshow(im1)
plt.show()