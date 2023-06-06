import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wire
import algorithms_old
import cv2


img = cv2.imread('yack.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.flip(img, 0)
drawing = algorithms.LineApproxBruteForce(img)
drawing.optimize()

# drawing = algorithms.LineApproxBruteForce(img)
# drawing.plot_summary()
# drawing.plot_costFunc3d()
# drawing.plot_distFromLine()



# drawing.imshow_android()