import algorithms
import cv2

img = cv2.imread('./img/yack.jpeg', cv2.IMREAD_GRAYSCALE)
drawing = algorithms.LineApproxBruteForce(img)
drawing.optimize()