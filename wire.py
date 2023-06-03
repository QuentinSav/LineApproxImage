import numpy as np

class Wire:
    def __init__(self, shape):
        width, height = shape

        # Initialize two random points in the image
        p1 = [width * np.random.rand(), height * np.random.rand()]
        p2 = [width * np.random.rand(), height * np.random.rand()]

        # Compute the parameters of the line passing by the points
        self.a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = p1[1] - self.a * p1[0]

    def isin(self, i, j):
        # Returns if the wire of this instance pass through the pixel ij

        if (j > self.eval(i) and self.eval(i) > (j + 1)):
            return True

        elif (j > self.eval(i + 1) and self.eval(i + 1) > (j + 1)):
            return True

        elif (j < self.eval(i) and self.eval(i + 1) < (j + 1)):
            return True

        elif (j < self.eval(i + 1) and self.eval(i) < (j + 1)):
            return True

        else:
            return False

    def eval(self, x):
        # Returns the y value for the line palrameters of the instance
        return self.a * x + self.b
