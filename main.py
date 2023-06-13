import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wire
import algorithms
import cv2


img = cv2.imread('img/taj.jpeg', cv2.IMREAD_GRAYSCALE)
#img = cv2.imread('taj.jpeg')
img = cv2.flip(img, 0)
optimizer = algorithms.Optimizer(img)
line_approx = optimizer.run()
line_approx.show_end_result()



# drawing = algorithms.LineApproxBruteForce(img)
# drawing.plot_summary()
# drawing.plot_costFunc3d()
# drawing.plot_distFromLine()



# drawing.imshow_android()