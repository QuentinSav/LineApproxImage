import numpy as np
import matplotlib.pyplot as plt
import cv2

class LineApprox:
    """Parent class of the different algorithm"""

    def __init__(self, target_img, n_lines=2000):

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
        self.target_img = cv2.flip(1 - self.normalize(target_img), 0)

        # Initialization of reconstruction img
        self.recon_img = np.zeros([self.height, self.width])

        print('Target image size: width=', self.width, ', height=', self.height, )

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
        fig, ax_array = plt.subplots(1, 3, subplot_kw={'aspect': 1}, sharex=True, sharey=True)

        # Draw the target image
        ax_array[0].set_title("Target image")
        ax_array[0].imshow(self.target_img, cmap='Greys')
        ax_array[0].axis([0, self.width, 0, self.height])
        ax_array[0].set_xticks([])
        ax_array[0].set_yticks([])

        # Draw all the computed lines
        ax_array[1].set_title("Line approximation image")
        for line in self.lines:
            line.add_to_plot(ax_array[1], self.hyper_c)

        # Draw the pixelized reconstructed image
        ax_array[2].set_title("Reconstructed image")
        ax_array[2].imshow(self.recon_img, cmap='Greys')
        ax_array[2].axis([0, self.width, 0, self.height])
        ax_array[2].set_xticks([])
        ax_array[2].set_yticks([])

        plt.show()


class LineApproxBruteForce(LineApprox):
    """Child class of the LineApprox that uses a brute force method to compute the approximation"""
    def optimize(self, n_random=100):
        """Function used to launch the optimization of the approximated image"""

        for i in range(self.n_lines):

            # Display the progress
            if i and not i % 100:
                print("Iteration ", i, "/", self.n_lines)

            # Initialize an empty batch of line and an empty array for the costs
            batch_lines = []
            cost_batch_lines = np.zeros([n_random])

            # Create a list of 100 randomly initialized lines
            for k in range(n_random):
                # t0 = time.time()

                # Initialize a random line
                line = Line([self.height, self.width])

                # Reconstruct the line
                recon_line = line.reconstruct_line(self.hyper_c)

                # Compute the cost of the line
                cost_batch_lines[k] = self.compute_cost(self.recon_img + recon_line)

                # Save the line in the list
                batch_lines.append(line)

            # Find the line with the smallest cost
            best_line = batch_lines[np.argmin(cost_batch_lines)]
            self.costs.append(np.min(cost_batch_lines))

            # Update reconstruction
            self.update_recon_img(best_line)

        self.show_end_result()


class Line:
    def __init__(self, shape):
        self.height, self.width = shape

        # Initialize two random points in the image
        p1 = [self.width * np.random.rand(), self.height * np.random.rand()]
        p2 = [self.width * np.random.rand(), self.height * np.random.rand()]

        # Compute the parameters of the line passing by the points
        self.a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        self.b = p1[1] - self.a * p1[0]

    def eval(self, x):
        # Returns the y value for the line parameters of the instance
        return self.a * x + self.b

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
            y0 = round(self.b)
            y1 = round(self.width * self.a + self.b)
            x0 = 0
            x1 = self.width

            # Draw the line
            recon_line = np.zeros([self.height, self.width])
            recon_line = cv2.line(recon_line, (x0, y0), (x1, y1), (1, 1, 1), 1)

        else:
            # Cover the case if an unknown method is passed as parameter
            print("Unknown method: Returning empty image")
            recon_line = np.zeros([self.height, self.width])

        return recon_line

    def add_to_plot(self, ax, c):
        # Compute the value of the line on the left and right border of the image
        x = np.linspace(0, self.width, 2)
        y = self.eval(x)

        # Plot the line on the existing axe
        ax.plot(x, y, color='black', linewidth=c / 5)
