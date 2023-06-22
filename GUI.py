import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import cv2

import analyze_interaction



def convert_from_traj_to_matrix(traj_npy_file):
    """
    reads a trajectory file and converts it to a 3D matrix, where each frame is a 2D matrix, and the pixels are the values in the trajectory.
    the 2 first columns in the trajectory are the x,y coordinates of the first particle, and the next 2 columns are the x,y coordinates of the second particle.
    do transformations to make the matrix look good as a GIF.
    :param traj_npy_file:
    :return:
    """
    data = np.load(traj_npy_file)
    # move the x coordinates 10 to the right, and the y coordinates 10 to the up:
    data[:, 0] += 10
    data[:, 1] += 10
    data[:, 2] += 10
    data[:, 3] += 10
    # multiply the coordinates by 1000 and then make them integers:
    data = data * 10
    data = data.astype(int)
    trajectory1 = data[:, 0:2]
    trajectory2 = data[:, 2:4]
    # create a 3D matrix with the size of the image, and the frames are the trajectory len (first dimension):
    np_arr = np.zeros((len(data), 201, 201))
    # for each frame in trajectory 1, insert 1 in the matrix in the coordinates of the particle:
    for i, frame in enumerate(trajectory1):
        np_arr[i, frame[0], frame[1]] = 1
    # for each frame in trajectory 2, insert 2 in the matrix in the coordinates of the particle:
    for i, frame in enumerate(trajectory2):
        np_arr[i, frame[0], frame[1]] = 2
    return np_arr

def show_gif_with_colors(np_arr, name="script_i"):
    """
    gets numpy array, that the 1st dimension is time (frames) and the other 2 dimensions are the pixels in the image.
    each pixel can be 0 (black), 1 (green) or 2 (red).
    :param np_arr:
    :return: shows a GIF from the frames, with the colors
    """
    for frame in np_arr:
        frame = frame.astype(np.uint8)
        frame[frame == 0] = 0
        frame[frame == 1] = 255
        frame[frame == 2] = 127
        cv2.imshow(name, frame)
        cv2.waitKey(100)
    cv2.destroyAllWindows()


def script1(np_arr):
    # create 3D matrix in numpy, that the 1st dimension is time (frames) and the other 2 dimensions are the pixels in the image. each pixel can be 0 (black), 1 (green) or 2 (red).
    # the 3D matrix is a numpy array of shape (100, 256, 256)
    np_arr = np.random.randint(0, 3, (100, 256, 256))
    return np_arr


def from_numpy_to_gif(np_arr, name="script_i"):
    """
    Gets numpy array with 3 dimensions, and saves it as an RGB GIF file.
    The 1st dimension is time (frames) and the other 2 dimensions are the pixels in the image.
    The pixels are one-dimensional but represent colors: 0 (black), 1 (green), or 2 (red).
    :param np_arr: numpy array
    :param name: name of the output GIF file
    :return:
    """
    np_arr = np_arr.astype(np.uint8)
    height, width = np_arr.shape[1:]  # Get height and width of the frames
    rgb_arr = np.zeros((np_arr.shape[0], height, width, 3), dtype=np.uint8)  # Create RGB array

    # Convert black, green, and red indices to RGB colors
    rgb_arr[np_arr == 0] = [0, 0, 0]    # Black
    rgb_arr[np_arr == 1] = [0, 255, 0]  # Green
    rgb_arr[np_arr == 2] = [255, 0, 0]  # Red

    frames = [Image.fromarray(frame, 'RGB') for frame in rgb_arr]
    frames[0].save(f'{name}.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
    return f'{name}.gif'

# Function to show multiple outputs side by side
def show_outputs(outputs):
    num_outputs = len(outputs)
    fig, axes = plt.subplots(1, num_outputs, figsize=(12, 4))
    for i in range(num_outputs):
        output = outputs[i]
        if isinstance(output, str):  # Handle GIF output
            gif_label = GIFLabel(root, output, f"{i}'th GIF:")
            # set title to the GIF
            gif_label.title.pack(side="left")
            gif_label.pack(side="left")
        else:  # Handle plot output
            axes[i].imshow(output)
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Function to show plots
def show_plots(data):
    # Generate and display plots using matplotlib
    plt.figure()
    plt.plot(data)
    plt.show()

# Custom Label widget to display GIFs
class GIFLabel(tk.Label):
    def __init__(self, master, gif_path, gif_title=""):
        super().__init__(master)
        self.gif_path = gif_path
        self.load_gif()
        self.animate()
        self.title = tk.Label(master, text=gif_title)

    def load_gif(self):
        self.gif = Image.open(self.gif_path)
        self.gif_frames = []
        try:
            while True:
                self.gif_frames.append(ImageTk.PhotoImage(self.gif.copy()))
                self.gif.seek(len(self.gif_frames))  # Move to the next frame
        except EOFError:
            pass

    def animate(self, frame_index=0):
        self.config(image=self.gif_frames[frame_index])
        frame_index = (frame_index + 1) % len(self.gif_frames)
        self.after(100, self.animate, frame_index)  # Adjust the delay (in milliseconds) between frames


# Function to run the scripts and generate outputs
def run_scripts(input_file):
    # Load the input data from the numpy file
    data = np.load(input_file)

    # Run script 1
    # Perform operations on data and generate output1
    output1 = script1(data)
    up_res = from_numpy_to_gif(output1, name="up_resolution")
    original = from_numpy_to_gif(data, name="original")
    converted = from_numpy_to_gif(convert_from_traj_to_matrix("/Users/michaleldar/Documents/year3/3DStructure/3DBioHackaton/X_high_resolution.npy"), name="converted")



    # Run script 2
    # Perform operations on data and generate output2

    # Run script 3
    # Perform operations on data and generate output3

    # Show the generated outputs
    # show_plots(data)

    # show_output(data, name="original")



    show_outputs(["/Users/michaleldar/Documents/year3/3DStructure/3DBioHackaton/7IsD.gif", original, up_res, converted])


# Function to show video/gif output
def show_output(output, name="script_i"):
    # Display the output as a video/gif using OpenCV or other libraries
    show_gif_with_colors(output, name=name)


# Function to show plots
def show_plots(data):
    # Generate and display plots using matplotlib
    plt.figure()
    plt.plot(data)
    plt.show()

# Create the GUI
root = tk.Tk()
root.title("The Virtual Microscope")
# set the window size to full screen
root.state('zoomed')


# Function to handle the button click event
def browse_file():
    # Open file dialog to select the input file
    file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")])
    if file_path:
        run_scripts(file_path)

# Create and position the GUI components
browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack()

# Function to handle the button click event
def get_params():
    # Open file dialog to select the input file
    file_path = filedialog.askopenfilename(filetypes=[("Numpy Files", "*.npy")])
    if file_path:
        data = np.load(file_path)
        trajectory1 = data[:70000:, 0:2]
        trajectory2 = data[:70000:, 2:4]
        distances = np.linalg.norm(trajectory1 - trajectory2, axis=1)
        k, R = analyze_interaction.estimate_K_and_R(trajectory1, trajectory2)
        # print the results on the screen with tkinter
        tk.Label(root, text=f"spring constant (K) = {k}").pack()
        tk.Label(root, text=f"Spring distance at rest (R_0) = {R}").pack()


# Create and position the GUI components
browse_button = tk.Button(root, text="Get Params", command=get_params)
browse_button.pack()

# Start the GUI event loop
root.mainloop()


# create 3D matrix in numpy and save in to numpy file:
# np_arr = np.random.randint(0, 255, (100, 256, 256))
# np.save("3D_matrix.npy", np_arr)

# load numpy file:
np_arr = np.load("3D_matrix.npy")

# show numpy array as GIF, each frame is a 2D matrix, and the pixels are the values in the matrix:
def show_gif(np_arr, name="script_i"):
    for frame in np_arr:
        cv2.imshow(name, frame/frame.max())
        cv2.waitKey(100)
    cv2.destroyAllWindows()

# show_gif(np_arr)
