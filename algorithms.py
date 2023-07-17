import numpy as np
import matplotlib.pyplot as plt
import cv2
from functools import wraps
import time
from skimage.transform import hough_line, hough_line_peaks

def compute_time(func):
    @wraps(func)
    def compute_time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return compute_time_wrapper


def rgb_to_cmyk(img):
    k = 1 - np.max(img, axis=2)
    c = (1-img[..., 2] - k)/(1-k)
    m = (1-img[..., 1] - k)/(1-k)
    y = (1-img[..., 0] - k)/(1-k)

    img_cmyk = (np.dstack((c, m, y, k)) * 255).astype(np.uint8)

    return img_cmyk


class LineApprox:
    """Parent class of the different algorithm"""

    def __init__(self, target_img, n_lines, color):

        self.color = color

        # Hyperparameters
        self.hyper_c = 0.1

        # List of optimized lines
        self.lines = []
        self.costs = []

        # Image dimensions
        self.height, self.width, = target_img.shape

        # Total number of lines
        self.n_lines = n_lines

        # Invert, normalize and flip image
        self.target_img = np.flipud(1- self.normalize(target_img))

        # Initialization of reconstruction img
        self.recon_img = np.zeros([self.height, self.width])

    def normalize(self, img):
        """Function that returns the normalized image"""

        # Get the min and max pixel intensity in the image
        max_value = np.max(img)
        min_value = np.min(img)

        # Check that the image is not composed only of zeros
        if img.any():
            normalized_img = (img - min_value) / (max_value - min_value)

        else:
            normalized_img = img

        # Return the normalized image
        return normalized_img

    def compute_cost(self, recon_img=np.empty):
        """Computes the squared error cost between an approximation and an image"""

        # Uses the instance reconstructed image if no image is passed as parameter
        if ~recon_img.any():
            recon_img = self.recon_img

        # Return the mean of the sum of squared errors
        return np.sum(np.square(self.target_img - self.normalize(recon_img)))

    def update_recon_img(self, best_line):
        """Function that updates the best line list and the current reconstruction image"""
        
        # Add the best line to the list of lines
        self.lines.append(best_line)

        # Update the reconstructed image with the new line
        self.recon_img = self.recon_img + best_line.reconstruct_line(self.hyper_c)

    def show_end_result(self):
        """Function to plot the final result"""

        # Creates figure and axes
        fig, ax_array = plt.subplots(1, 2, subplot_kw={'aspect': 1}, sharex=True, sharey=True)

        # Draw the target image

        ax_array[0].set_title("Target image")
        ax_array[0].imshow(self.target_img, cmap='Greys')
        ax_array[0].axis([0, self.width, 0, self.height])
        ax_array[0].set_xticks([])
        ax_array[0].set_yticks([])

        fig.show()

        # Draw all the computed lines

        ax_array[1].set_title("Line approximation image")

        for line in self.lines:
            if self.color == 'greyscale':
                color = 'black'
            else:
                color = self.color

            line.add_to_plot(ax_array[1], self.hyper_c, color)


        # Draw the pixelized reconstructed image
        fig = plt.figure()

        plt.title("Reconstructed image")
        plt.plot(self.costs)
        plt.axis([0, self.n_lines, 0, np.max(self.costs)])
        plt.ylabel("Cost")
        plt.xlabel("Iteration")

        plt.show()


class LineApproxColor:
    def __init__(self, line_approx_list):
        if len(line_approx_list) == 3:
            self.r_line_approx = line_approx_list[0]
            self.g_line_approx = line_approx_list[1]
            self.b_line_approx = line_approx_list[2]

        elif len(line_approx_list) == 4:
            self.c_line_approx = line_approx_list[0]
            self.m_line_approx = line_approx_list[1]
            self.y_line_approx = line_approx_list[2]
            self.k_line_approx = line_approx_list[3]

        else:
            raise ValueError("Too many channels, cannot interpret as RGB image or CMYK.")



    def show_end_result_rgb(self):

        fig, ax_array = plt.subplots(1, 3, subplot_kw={'aspect': 1}, sharex=True, sharey=True)

        ax_array[0].set_title("Target image")
        ax_array[0].imshow(self.r_line_approx.target_img, cmap='Greys')
        ax_array[0].axis([0, self.r_line_approx.width, 0, self.r_line_approx.height])
        ax_array[0].set_xticks([])
        ax_array[0].set_yticks([])

        fig.show()

        ax_array[1].set_title("Line approximation image")

        for line_approx in [self.r_line_approx, self.g_line_approx, self.b_line_approx]:
            for line in line_approx.lines:
                if line_approx.color == 'greyscale':
                    color = 'black'
                else:
                    color = line_approx.color

                line.add_to_plot(ax_array[1], self.r_line_approx.hyper_c, color)

        ax_array[2].set_title("Reconstructed image")
        ax_array[2].imshow(self.r_line_approx.recon_img, cmap='Greys')
        ax_array[2].axis([0, self.r_line_approx.width, 0, self.r_line_approx.height])
        ax_array[2].set_xticks([])
        ax_array[2].set_yticks([])

        plt.show()


    def show_end_result_cmyk(self):
        
        fig, ax_array = plt.subplots(1, 3, subplot_kw={'aspect': 1}, sharex=True, sharey=True)

        ax_array[0].set_title("Target image")
        ax_array[0].imshow(self.c_line_approx.target_img, cmap='Greys')
        ax_array[0].axis([0, self.c_line_approx.width, 0, self.c_line_approx.height])
        ax_array[0].set_xticks([])
        ax_array[0].set_yticks([])

        fig.show()

        ax_array[1].set_title("Line approximation image")

        for line_approx in [self.c_line_approx, self.m_line_approx, self.y_line_approx, self.k_line_approx]:
            for line in line_approx.lines:
                if line_approx.color == 'greyscale':
                    color = 'black'
                else:
                    color = line_approx.color

                line.add_to_plot(ax_array[1], self.c_line_approx.hyper_c, color)

        ax_array[2].set_title("Reconstructed image")
        ax_array[2].imshow(self.c_line_approx.recon_img, cmap='Greys')
        ax_array[2].axis([0, self.c_line_approx.width, 0, self.c_line_approx.height])
        ax_array[2].set_xticks([])
        ax_array[2].set_yticks([])

        plt.show()

        
class Optimizer:
    def __new__(cls, img, n_lines=5000):
        if len(img.shape) == 2:
            return OptimizerGreyscale(img, n_lines, color='greyscale')

        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                print('Image type = ''rgb''')
                return OptimizerColor(img, n_lines, color_type='rgb')
            
            elif img.shape[2] == 4:
                print('Image type = ''cmyk''')
                return OptimizerColor(img, n_lines, color_type='cmyk')

        else:
            raise ValueError("Unsupported image format")


class OptimizerGreyscale:
    #@compute_time
    def __init__(self, target_img, n_lines, color):
        self.line_approx = LineApprox(target_img, n_lines, color)

    #@compute_time
    def run(self, n_random:int=100):
        """Function used to launch the optimization of the approximated image"""

        for i in range(self.line_approx.n_lines):

            # Display the progress
            if i and not (i + 1) % 100:
                print("Iteration ", (i + 1), "/", self.line_approx.n_lines)

            # Initialize an empty batch of line and an empty array for the costs
            batch_lines = []
            cost_batch_lines = np.zeros([n_random])

            # Create a list of 100 randomly initialized lines
            for k in range(n_random):
                # Initialize a random line
                line = Line([self.line_approx.height, self.line_approx.width])

                # Reconstruct the line
                recon_line = line.reconstruct_line(self.line_approx.hyper_c)

                # Compute the cost of the line
                cost_batch_lines[k] = self.line_approx.compute_cost(self.line_approx.recon_img + recon_line)

                # Save the line in the list
                batch_lines.append(line)

            # Find the line with the smallest cost
            best_line = batch_lines[np.argmin(cost_batch_lines)]
            self.line_approx.costs.append(np.min(cost_batch_lines))

            # Update reconstruction
            self.line_approx.update_recon_img(best_line)

        return self.line_approx
            
    @compute_time
    def run_hough_transform(self):
        
        for i in range(self.line_approx.n_lines):

            # Display the progress
            if i and not (i + 1) % 1:
                print("Iteration ", (i + 1), "/", self.line_approx.n_lines)

            # Initialize an empty batch of line and an empty array for the costs
            batch_lines = []

            H = self.radon_transform()
            
            index_r, index_theta  = np.unravel_index(H.argmax(), H.shape)
            best_theta = theta[index_theta]
            best_r = r[index_r]
            
            # Reconstruct the line
            self.line_approx.update_recon_img(line)

        return self.line_approx

    @compute_time
    def hough_transform(self, n_theta=100, n_r=50):

        theta = np.linspace(0, 2*np.pi, n_theta)
        r = np.linspace(0, np.sqrt(np.square(self.line_approx.height) + np.square(self.line_approx.width)), n_r)
        H = np.zeros([n_r, n_theta])
        target_img = self.line_approx.target_img - self.line_approx.recon_img

        x = np.arange(self.line_approx.width)
        y = np.arange(self.line_approx.height)

        X, Y = np.meshgrid(x, y)

        distances = X * np.cos(theta) + Y * np.sin(theta) - r[:, np.newaxis, np.newaxis]
        denominator = self.line_approx.hyper_c + np.square(distances)
        H = np.sum(target_img * self.line_approx.hyper_c / denominator)

        index_r, index_theta  = np.unravel_index(H.argmax(), H.shape)
        best_theta = theta[index_theta]
        best_r = r[index_r]

        return Line([self.line_approx.height, self.line_approx.width], theta=best_theta, r=best_r)

    def radon_transform(self, n_theta=100, n_rho=50):
        theta_array = np.linspace(1e-3, np.pi, n_theta)
        
        diag = np.sqrt(np.square(self.line_approx.height) + np.square(self.line_approx.width))
        rho_array = np.linspace(-diag, diag, n_rho)

        g_radon = np.zeros([n_rho, n_theta])

        for t in range(n_theta):
            theta = theta_array[t]
            x_min = -(self.line_approx.width)/2
            y_min = -(self.line_approx.height)/2
            delta_x = 1
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            rho_offset = x_min*(cos_theta + sin_theta)

            if sin_theta > 1/np.sqrt(2):
                alpha = - cos_theta/sin_theta
                for r in range(n_rho):
                    rho = rho_array[r]

                    beta = (rho - rho_offset)/(delta_x*sin_theta)
                    if alpha > 0:
                        m_min = np.max([0, np.ceil(-(beta+1/2)/alpha)])
                        m_max = np.min([self.line_approx.width - 1, np.floor((self.line_approx.height - 1/2 - beta)/alpha)])
            
                    else:
                        m_min = np.max([0, np.ceil((self.line_approx.height - 1/2 - beta)/alpha)])
                        m_max = np.max([self.line_approx.width - 1, np.floor(-(beta+1/2)/alpha)])

                    sum = 0

                    m = np.arange(m_min, m_max)
                    g_radon[r, t] = delta_x*np.sum(self.line_approx.target_img[(m).astype('uint16'), (np.round(alpha*m + beta)).astype('uint16')])/sin_theta
            
            else:
                alpha = - sin_theta/cos_theta
                for r in range(n_rho):
                    rho = rho_array[r]

                    beta = (rho - rho_offset)/(delta_x*cos_theta)
                    if alpha > 0:
                        n_min = np.max([0, np.ceil(-(beta+1/2)/alpha)])
                        n_max = np.min([self.line_approx.height - 1, np.floor((self.line_approx.width - 1/2 - beta)/alpha)])
            
                    else:
                        n_min = np.max([0, np.ceil((self.line_approx.width - 1/2 - beta)/alpha)])
                        n_max = np.max([self.line_approx.height - 1, np.floor(-(beta+1/2)/alpha)])

                    sum = 0

                    n = np.arange(n_min, n_max)
                    g_radon[r, t] = delta_x*np.sum(self.line_approx.target_img[(np.round(alpha*n - beta)).astype('uint16'), (n).astype('uint16')])/np.abs(cos_theta)
            
        return g_radon

class OptimizerColor(OptimizerGreyscale):
    #@compute_time
    def __init__(self, img, n_lines, color_type):
        self.img = img
        self.n_lines = n_lines
        self.line_approx_list = []

        if color_type == 'rgb':
            self.colors = ['red', 'green', 'blue']

        elif color_type == 'cmyk':
            self.colors = ['cyan', 'magenta', 'yellow', 'black']

    #@compute_time
    def run(self):
        
        self.compute_color_ratio()

        for k in range(len(self.colors)):
            super().__init__(self.img[:, :, k], self.n_lines_colors[k], color=self.colors[k])
            self.line_approx_list.append(super().run())

        line_approx_color = LineApproxColor(self.line_approx_list)

        return line_approx_color

    def compute_color_ratio(self):
        
        sum_colors = np.zeros(len(self.colors))

        for k in range(len(self.colors)):
            sum_colors[k] = np.sum(self.img[..., k])

        weight_colors = sum_colors/np.sum(sum_colors)
        print(weight_colors)
        self.n_lines_colors = (self.n_lines * weight_colors).astype('uint')



class Line:
    #@compute_time
    def __init__(self, shape):
        self.height, self.width = shape

        # Initialize two random points in the image
        p1 = [self.width * np.random.rand(), self.height * np.random.rand()]
        p2 = [self.width * np.random.rand(), self.height * np.random.rand()]

        # Compute the parameters of the line passing by the points
        self.a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = p1[1] - self.a * p1[0]

    #def __init__(self, shape, theta, r):
    #    self.height, self.width = shape
    #    self.theta = theta
    #    self.r = r
    #    
    #    self.a = - np.cos(theta)/np.sin(theta)
    #    self.b = r/np.sin(theta)
        

    #@compute_time
    def eval(self, x):
        # Returns the y value for the line parameters of the instance
        return self.a * x + self.b

    #@compute_time
    def reconstruct_line(self, c, method='binary'):
        """Reconstruct a line in the image space"""

        if method == 'gaussian':
            # Create a meshgrid to compute values in plane more efficiently
            height_arr, width_arr = np.meshgrid(range(self.height), range(self.width))

            # Computes the gaussian function
            dist_from_line = (self.a * width_arr - height_arr + self.b) / np.sqrt(np.square(self.a) + 1)
            recon_line = np.exp(-np.square(dist_from_line) / (2 * np.square(c)))

        elif method == 'binary':
            # Compute the starting and ending points of the line
            eps = 1e-1

            if self.a > eps and self.a < 2*np.pi -eps :
                y0 = round(self.b)
                y1 = round(self.width * self.a + self.b)
                x0 = 0
                x1 = self.width

            else:
                y0 = 0 
                y1 = self.height
                x0 = round(-self.b/self.a)
                x1 = round((self.height - self.b)/self.a)

            # Draw the line
            recon_line = np.zeros([self.height, self.width])
            recon_line = cv2.line(recon_line, (x0, y0), (x1, y1), (1, 1, 1), 1)

        else:
            # Cover the case if an unknown method is passed as parameter
            print("Unknown method: Returning empty image")
            recon_line = np.zeros([self.height, self.width])

        return recon_line


    def add_to_plot(self, ax, c, color):

        # Compute the value of the line on the left and right border of the image
        x = np.linspace(0, self.width, 2)
        y = self.eval(x)

        # Plot the line on the existing axe

        ax.plot(x, y, color=color, linewidth=c/5)
