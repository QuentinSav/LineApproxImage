import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wire
import line_approx


img = Image.open('image.jpeg').convert('L')

drawing = line_approx.Drawing(img, n_wires=1)
# drawing.plot_summary()
# drawing.plot_costFunc3d()
# drawing.plot_distFromLine()
drawing.optimize()
drawing.imshow_android()