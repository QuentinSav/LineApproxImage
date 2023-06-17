import algorithms
import cv2

img = cv2.imread('./img/oldman.jpeg', cv2.IMREAD_GRAYSCALE)
drawing = algorithms.LineApprox(img)
drawing.optimize()
