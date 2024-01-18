import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from absl import logging
from urllib import request

# Set the logging verbosity level to ERROR to suppress unnecessary TensorFlow messages
logging.set_verbosity(logging.ERROR)

class VideoActionRecognitionApp:
    def __init__(self, root):
        # Initialize the VideoActionRecognitionApp with the given root (Tkinter window)
        self.root = root
        self.root.title("Video Action Recognition App")  # Set the window title

        # Initialize the video_path variable to store the selected video file path
        self.video_path = ""

        # Load the Inflated 3D CNN model from TensorFlow Hub (I3D model for action recognition)
        self.i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

        # Get the kinetics-400 labels (action labels for the I3D model)
        self.labels_i3d = self.fetch_kinetics_labels()

        # Create GUI components (widgets) using the create_widgets method
        self.create_widgets()

    def fetch_kinetics_labels(self):
        # Fetch the kinetics-400 action labels from a URL and return them as a list
        kinetics_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
        with request.urlopen(kinetics_url) as obj:
            labels = [line.decode("utf-8").strip() for line in obj.readlines()]
        return labels

    def create_widgets(self):
        # Create GUI components (widgets) for the application

        # Video selection button
        self.select_video_button = tk.Button(self.root, text="Select Video", command=self.select_video)
        self.select_video_button.pack(pady=10)

        # Selected video label (displays the path of the selected video)
        self.selected_video_label = tk.Label(self.root, text="")
        self.selected_video_label.pack(pady=5)

        # Start recognition button
        self.recognize_button = tk.Button(self.root, text="Start Recognition", command=self.start_recognition)
        self.recognize_button.pack(pady=10)

        # Label for displaying recognized action
        self.action_label = tk.Label(self.root, text="Recognized Action: ")
        self.action_label.pack(pady=10)

    def select_video(self):
        # Callback method for the "Select Video" button
        # Opens a file dialog to select a video file and updates the selected video label
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if self.video_path:
            self.selected_video_label.config(text=f"Selected Video: {self.video_path}")

    def load_video(self, path, max_frames=0, resize=(224, 224)):
        # Load video frames from the specified video file path
        # Optionally resize frames and limit the number of frames loaded (max_frames)

        # Initialize a video capture object using OpenCV
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize the frame (optional) and convert color channels to RGB
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]

                # Append the frame to the frames list
                frames.append(frame)

                # Break if the maximum number of frames is reached (if max_frames > 0)
                if max_frames > 0 and len(frames) == max_frames:
                    break
        except Exception as e:
            print(f"Error loading video: {e}")
        finally:
            cap.release()  # Release the video capture object
        return np.array(frames) / 255.0  # Normalize pixel values to [0, 1]

    def start_recognition(self):
        # Callback method for the "Start Recognition" button
        # Initiates the action recognition process on the selected video
        if not self.video_path:
            print("Please select a video first.")
            return

        # Load video frames from the selected video
        video = self.load_video(self.video_path)

        # Call the predict_i3d method to perform action recognition using the I3D model
        self.predict_i3d(video)

    def predict_i3d(self, sample_video):
        # Perform action recognition on the input video frames using the I3D model

        # Add a batch axis to the sample video.
        model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

        # Make predictions using the I3D model
        logits = self.i3d(model_input)['default'][0]
        probabilities = tf.nn.softmax(logits)

        # Get the action label with the highest probability
        recognized_action = self.labels_i3d[np.argmax(probabilities)]

        # Update the action_label widget to display the recognized action
        self.action_label.config(text=f"Recognized Action: {recognized_action}")


if __name__ == "__main__":
    # Main execution block

    # Create the Tkinter root window
    root = tk.Tk()

    # Create an instance of the VideoActionRecognitionApp class
    app = VideoActionRecognitionApp(root)

    # Start the Tkinter event loop
    root.mainloop()
