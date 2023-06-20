import algorithms
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./img/oldman.jpg')#, cv2.IMREAD_GRAYSCALE)

# To be implemented
img = algorithms.rgb_to_cmyk(img)


optimizer = algorithms.Optimizer(img, n_lines=5000)
line_approx = optimizer.run()

line_approx.show_end_result_cmyk()
