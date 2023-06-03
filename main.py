import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wire
import algorithms


img = Image.open('cat.jpeg').convert('L')

drawing = algorithms.LineApproxBruteForce(img)
# drawing.plot_summary()
# drawing.plot_costFunc3d()
# drawing.plot_distFromLine()
drawing.optimize()


# drawing.imshow_android()