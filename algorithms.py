import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class LineApprox:

    def __init__(self, target_img, n_lines=1000):

        # Hyperparameters
        self.c = 0.1

        # List of optimized lines
        self.lines = []

        # Image dimensions
        self.m = target_img.width
        self.n = target_img.height

        # Total number of lines
        self.n_lines = n_lines

        self.target_img = 1-self.normalize(target_img)

        # Initialization of reconstruction img
        self.recon_img = np.zeros([self.n, self.m])

        print('Target image size ------------------')
        print('n=', self.n, ', m=', self.m)

    def reconstruct(self, line, method='gaussian'):

        # Initialize reconstructed image
        recon_line = np.zeros([self.n, self.m])

        if method == 'gaussian':
            for i in range(self.m):
                for j in range(self.n):
                    recon_line[j, i] = line.eval_point_in_plane(i, j, self.c)

        elif method == 'binary':
            for i in range(self.m):
                for j in range(self.n):
                    if line.isin(i, j):
                        recon_line[j, i] = 1

        return recon_line

    def normalize(self, img):
        max_value = np.max(img)
        min_value = np.min(img) - 1e-3

        return (img - min_value) / (max_value - min_value)

    def compute_cost(self, recon_img=np.empty):

        # Uses the instance reconstructed image if no image is passed as parameter
        if ~recon_img.any():
            recon_img = self.recon_img

        # Return the mean of the sum of squared errors
        return 1/(self.m*self.n) * np.sum((self.target_img - self.normalize(recon_img))**2)

    def update_recon_img(self, best_line):
        self.lines.append(best_line)
        self.recon_img = self.recon_img + self.reconstruct(best_line)

    def show_end_result(self):
        f = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Target image")
        plt.imshow(self.target_img, cmap='Greys')
        plt.axis([0, self.m, 0, self.m])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed image")
        plt.imshow(self.recon_img, cmap='Greys')
        plt.axis([0, self.m, 0, self.m])
        plt.xticks([])
        plt.yticks([])

        plt.show()

    def show_live_recon(self):
        f = plt.figure(1)
        plt.title("Target image")
        plt.imshow(self.target_img, cmap='Greys')
        plt.axis([0, self.m, 0, self.m])
        plt.xticks([])
        plt.yticks([])
        plt.show()

        return f

class LineApproxBruteForce(LineApprox):
    def optimize(self, n_random=100):

        for i in range(self.n_lines):

            batch_lines = []
            cost_batch_lines = np.zeros([n_random])

            #f = self.show_live_recon()

            # Create a list of 100 randomly initialized lines
            for k in range(n_random):

                line = Line([self.m, self.n])
                batch_lines.append(line)
                recon_line = self.reconstruct(line, method='gaussian')
                cost_batch_lines[k] = self.compute_cost(self.recon_img+recon_line)

            # Find the line with the smallest cost
            best_line = batch_lines[np.argmin(cost_batch_lines)]
            # best_line.add_to_plot(f, self.n, self.m)

            # Update reconstruction
            self.update_recon_img(best_line)

            print("Iteration ", i, "/", self.n_lines)

        self.show_end_result()


def plot_costFunc3d(self):

    # Initialize
    a_len = 21
    b_len = 21
    a = np.linspace(-10, 10, a_len)
    b = np.linspace(-10, 10, b_len)

    # a = [-3]
    # b = [5]
    # a_len = 1
    # b_len = 1

    cost = np.zeros([a_len, b_len])
    dist_pix = np.zeros([a_len, b_len])
    inv_dist_weight_pix = np.zeros([a_len, b_len])

    print('Cost ----------------------------------------')

    # Iterates the parameters
    for a_idx in range(a_len):
        for b_idx in range(b_len):
            print('- - - - - - - - - -')
            print('a =', a[a_idx])
            print('b =', b[b_idx])

            # Iterates the pixel
            for i in range(self.m):
                for j in range(self.n):
                    dist_pix = (a[a_idx] * i - j + b[b_idx]) / (a[a_idx] ** 2 + 1) ** (1 / 2)
                    inv_dist_weight_pix = np.exp(-dist_pix ** 2 / (2 * self.hyp_c ** 2))
                    cost_pix = (self.target_img[j, i] - inv_dist_weight_pix) ** 2

                    cost[a_idx, b_idx] = cost[a_idx, b_idx] + cost_pix

                if False:
                    print('- - - - - - - - - - ')
                    print('target pixel =', self.target_img[0, i])
                    print('distance      =', dist_pix)
                    print('weight         =', inv_dist_weight_pix)
                    print('cost              =', cost_pix)

            if True:
                print('cost =', cost[a_idx, b_idx])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    A, B = np.meshgrid(a, b)
    surf = ax.plot_surface(A.T, B.T, cost)

    plt.title("Cost function")
    plt.xlabel('a')
    plt.ylabel('b')
    plt.show()


def imshow_android(self):
    plt.subplot(3, 2, 1)
    plt.title("Target image")
    plt.imshow(self.target_img, cmap='Greys')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 2, 2).set_aspect('equal')
    plt.title("String image")
    x = np.linspace(0, self.m, 2)
    for line in self.lines:
        plt.plot(x, -line.eval(x), 'k', linewidth=.1)
    plt.axis([0, self.m, -self.n, 0])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 2, 3)
    plt.title("Grad a image")
    # plt.imshow(self.recon_img, cmap='Greys')
    plt.imshow(self.grad_a)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 2, 4)
    plt.title("Grad b image")
    # plt.imshow(self.recon_img, cmap='Greys')
    plt.imshow(self.grad_b)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 2, 5)
    plt.title("Cost image")
    plt.imshow(self.recon_img, cmap='Greys')
    # plt.imshow(self.cost_pix, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplot(3, 2, 6)
    plt.title("grad evolution")
    plt.plot(self.array_grad_a)
    plt.plot(self.array_grad_b)
    # plt.plot(self.array_cost)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    print(self.array_grad_b)
    plt.show()

def plot_summary(self):

    self.target_img[10, 0] = 1
    plt.subplot(1, 1, 1)
    plt.title("Target image")
    plt.imshow(self.target_img, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.show()


class LineApproxGD:
    def __init__(self, target_img, n_lines=1):

        test_case = 2

        # List of optimized lines
        self.lines = []

        # Hyperparameters
        self.hyp_c = 1
        self.hyp_alpha = 1e-2

        # Image dimensions
        self.m = target_img.width
        self.n = target_img.height

        # Total number of lines
        self.n_lines = n_lines

        # Save target img in object and normalize
        if test_case == False:
            self.target_img = 255 - np.flip(np.array(target_img), axis=1)

        elif test_case == 1:
            self.m = 2
            self.n = 2
            self.target_img = np.zeros([self.m, self.n])
            self.target_img[0, 1] = 255
            self.target_img[1, 0] = 255

        elif test_case == 2:
            self.m = 10
            self.n = 10
            self.target_img = np.zeros([self.m, self.n])
            a = -3
            b = 5

            for i in range(self.m):
                for j in range(self.n):
                    dist_pix = (a * i - j + b) / (a ** 2 + 1) ** (1 / 2)
                    self.target_img[j, i] = np.exp(-dist_pix ** 2 / (2 * (self.hyp_c / 2) ** 2))

        self.target_img = self.normalize(self.target_img)

        # Initialization of reconstruction img
        self.recon_img = np.zeros([self.n, self.m])

        # Initialization of gradient img
        self.grad_a = np.zeros([self.n, self.m])
        self.grad_b = np.zeros([self.n, self.m])
        self.cost_pix = np.zeros([self.n, self.m])
        self.array_grad_a = []
        self.array_grad_b = []

        print('Target image size ------------------')
        print('n=', self.n, ', m=', self.m)
        # print(self.target_img)

    def reconstruct(self, line):
        recon_line = np.zeros([self.n, self.m])

        for i in range(self.m):
            for j in range(self.n):
                if line.isin(i, j):
                    recon_line[j, i] = 1
        return recon_line

    def optimize(self):

        for z in range(self.n_lines):
            line = line([self.m, self.n])

            line = self.gradient_descent(line)

            if line:
                self.lines.append(line)
                self.recon_img = self.recon_img + self.reconstruct(line)

    def gradient_descent(self, line):
        a = line.a
        b = line.b
        array_grad_a = np.array([1])
        array_grad_b = np.array([1])
        array_cost = np.array([1])

        prev_cost = 1e20
        cost = 1e19
        iter = 1
        print('a =', round(a, 2), ', b=', round(b, 2))

        while (prev_cost - cost) > 1e-3:
            prev_cost = cost
            grad_a, grad_b, cost = self.gradient(a, b)

            array_grad_a[:1] = grad_a
            array_grad_b[:1] = grad_b
            # array_cost[:1] = cost

            a = a - self.hyp_alpha * grad_a
            b = b - self.hyp_alpha * grad_b

            print('GD iteration #', iter, '-------')
            print('a =', round(a, 2), ', b=', round(b, 2))
            print('cost = ', round(cost, 2))
            print('grad(a) =', round(grad_a, 2), 'grad(b) =', round(grad_b, 2))
            iter = iter + 1

            if self.check_constraint(a, b):
                print('out of limit')
                return False

        self.array_grad_a = array_grad_a
        self.array_grad_b = array_grad_b
        #   self.array_cost = array_cost

        line.a = a
        line.b = b

        return line

    def check_constraint(self, a, b):
        if b <= 0 and a < -1 / self.m * b:
            return True

        if b >= self.n and a > -1 / self.m * b + self.n / self.m:
            return True

        else:
            return False

    def gradient(self, a, b):

        c = self.hyp_c
        y = self.normalize(self.target_img - self.recon_img)

        grad_a = 0
        grad_b = 0
        cost = 0
        cost_pix = 0
        dist_pix = 0

        for i in range(self.m):
            for j in range(self.n):
                self.grad_a[j, i] = (
                            -2 * y[j, i] * np.exp(-((a * i - j + b) / (2 * c ** 2 * (a ** 2 + 1) ** (1 / 2)))) * (-(
                                (i * (2 * c ** 2 * (a ** 2 + 1) ** (1 / 2))) - (a * i - j + b) * (
                                    2 * a * c ** 2 / (a ** 2 + 1) ** (1 / 2))) / (
                                                                                                                              4 * c ** 4 * a ** 2 + 4 * c ** 4)) + np.exp(
                        -(((a * i - j + b) / np.sqrt(a ** 2 + 1)) ** 2 / (c ** 2 * (a ** 2 + 1) ** (1 / 2)))) * (-(
                                (i * (2 * c ** 2 * (a ** 2 + 1) ** (1 / 2))) - (a * i - j + b) * (
                                    2 * a * c ** 2 / (a ** 2 + 1) ** (1 / 2))) / (c ** 4 * a ** 2 + c ** 4)))

                self.grad_b[j, i] = (
                            -2 * y[j, i] * np.exp(-((a * i - j + b) / (2 * c ** 2 * (a ** 2 + 1) ** (1 / 2)))) * (
                                -1 / (2 * c ** 2 * (a ** 2 + 1) ** (1 / 2))) + np.exp(
                        -(((a * i - j + b) / np.sqrt(a ** 2 + 1)) ** 2 / (c ** 2 * (a ** 2 + 1) ** (1 / 2)))) * (
                                        -1 / (c ** 2 * (a ** 2 + 1) ** (1 / 2))))

                self.cost_pix[j, i] = (y[j, i] - np.exp(
                    -(a * i - j + b) ** 2 / (2 * c ** 2 * (a ** 2 + 1) ** (1 / 2)))) ** 2

                dist_pix = (a * i - j + b) / (2 * c ** 2 * np.sqrt(a ** 2 + 1))

                cost = self.cost_pix[j, i] + cost
                grad_a = self.grad_a[j, i] + grad_a
                grad_b = self.grad_b[j, i] + grad_b

        return grad_a, grad_b, cost

    def normalize(self, img):
        max_value = np.max(img)
        min_value = np.min(img) - 1e-3

        return (img - min_value) / (max_value - min_value)

    def plot_distFromLine(self):

        dist_pix = np.zeros([self.n, self.m])
        inv_dist_weight_pix = np.zeros([self.n, self.m])
        a = 1
        b = 0
        # c = self.hyp_c
        c = 10
        N, M = np.meshgrid(range(self.m), range(self.n))

        # Iterates the pixel
        for i in range(self.m):
            for j in range(self.n):
                dist_pix[j, i] = (a * i - j + b) / (a ** 2 + 1) ** (1 / 2)
                inv_dist_weight_pix[j, i] = np.exp(-dist_pix[j, i] ** 2 / (2 * c ** 2))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(N, M, inv_dist_weight_pix)

        plt.title("Distance from line function")
        plt.show()

    def plot_costFunc3d(self):

        # Initialize
        a_len = 21
        b_len = 21
        a = np.linspace(-10, 10, a_len)
        b = np.linspace(-10, 10, b_len)

        # a = [-3]
        # b = [5]
        # a_len = 1
        # b_len = 1

        cost = np.zeros([a_len, b_len])
        dist_pix = np.zeros([a_len, b_len])
        inv_dist_weight_pix = np.zeros([a_len, b_len])

        print('Cost ----------------------------------------')

        # Iterates the parameters
        for a_idx in range(a_len):
            for b_idx in range(b_len):
                print('- - - - - - - - - -')
                print('a =', a[a_idx])
                print('b =', b[b_idx])

                # Iterates the pixel
                for i in range(self.m):
                    for j in range(self.n):
                        dist_pix = (a[a_idx] * i - j + b[b_idx]) / (a[a_idx] ** 2 + 1) ** (1 / 2)
                        inv_dist_weight_pix = np.exp(-dist_pix ** 2 / (2 * self.hyp_c ** 2))
                        cost_pix = (self.target_img[j, i] - inv_dist_weight_pix) ** 2

                        cost[a_idx, b_idx] = cost[a_idx, b_idx] + cost_pix

                    if False:
                        print('- - - - - - - - - - ')
                        print('target pixel =', self.target_img[0, i])
                        print('distance      =', dist_pix)
                        print('weight         =', inv_dist_weight_pix)
                        print('cost              =', cost_pix)

                if True:
                    print('cost =', cost[a_idx, b_idx])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        A, B = np.meshgrid(a, b)
        surf = ax.plot_surface(A.T, B.T, cost)

        plt.title("Cost function")
        plt.xlabel('a')
        plt.ylabel('b')
        plt.show()

    def imshow_android(self):
        plt.subplot(3, 2, 1)
        plt.title("Target image")
        plt.imshow(self.target_img, cmap='Greys')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 2, 2).set_aspect('equal')
        plt.title("String image")
        x = np.linspace(0, self.m, 2)
        for line in self.lines:
            plt.plot(x, -line.eval(x), 'k', linewidth=.1)
        plt.axis([0, self.m, -self.n, 0])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 2, 3)
        plt.title("Grad a image")
        # plt.imshow(self.recon_img, cmap='Greys')
        plt.imshow(self.grad_a)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 2, 4)
        plt.title("Grad b image")
        # plt.imshow(self.recon_img, cmap='Greys')
        plt.imshow(self.grad_b)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 2, 5)
        plt.title("Cost image")
        plt.imshow(self.recon_img, cmap='Greys')
        # plt.imshow(self.cost_pix, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        plt.subplot(3, 2, 6)
        plt.title("grad evolution")
        plt.plot(self.array_grad_a)
        plt.plot(self.array_grad_b)
        # plt.plot(self.array_cost)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        print(self.array_grad_b)
        plt.show()

    def plot_summary(self):
        self.target_img[10, 0] = 1
        plt.subplot(1, 1, 1)
        plt.title("Target image")
        plt.imshow(self.target_img, cmap='Greys')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.show()
        
        
        
class Line:
    def __init__(self, shape):
        width, height = shape

        # Initialize two random points in the image
        p1 = [width * np.random.rand(), height * np.random.rand()]
        p2 = [width * np.random.rand(), height * np.random.rand()]

        # Compute the parameters of the line passing by the points
        self.a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = p1[1] - self.a * p1[0]

    def isin(self, i, j):
        # Returns if the line of this instance pass through the pixel ij

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
        # Returns the y value for the line parameters of the instance
        return self.a * x + self.b

    def eval_point_in_plane(self, i, j, c):
        # Returns the value of a point in the plane for the line parameters of the instance (note: when the line is
        # modeled as a gaussian in the plane)

        # Distance between the point and the line
        dist_from_line = (self.a * i - j + self.b) / (self.a ** 2 + 1) ** (1 / 2)

        # Value of the point in the plane
        value_point = np.exp(-dist_from_line ** 2 / (2 * c ** 2))

        return value_point

    def add_to_plot(self, plot, m, n):
        # To be written

        plt.figure(1)

        x = np.linspace(0, m, 3)
        y = self.eval(x)
        plt.plot(x, y)

        plt.show(block = False)
        plt.pause(0.001)