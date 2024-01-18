# Video Action Recognition App

This is a simple Python application using Tkinter, OpenCV, and TensorFlow to perform action recognition on a selected video file. The app leverages the I3D (Inflated 3D ConvNet) model from TensorFlow Hub for action recognition.

## Prerequisites

Before running the application, make sure you have the following libraries installed:

- `tkinter`
- `opencv-python`
- `numpy`
- `tensorflow`
- `tensorflow-hub`

You can install them using the following command:

```bash
pip install tk opencv-python numpy tensorflow tensorflow-hub
```

## Usage

To use the Video Action Recognition App, follow these steps:

1. Run the application by executing the `main.py` script.
   ```bash
   python main.py
   ```

2. The app window will appear with the title "Video Action Recognition App."

3. Click the "Select Video" button to choose a video file (supported formats: `.mp4` and `.avi`).

4. After selecting the video, the path will be displayed below the button.

5. Click the "Start Recognition" button to initiate the action recognition process.

6. The recognized action label will be displayed below the "Recognized Action" label.

## Implementation Details

- The app uses Tkinter for the graphical user interface.
- The I3D model for action recognition is loaded from TensorFlow Hub.
- Kinetics-400 action labels are fetched from a specified URL.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- TensorFlow Hub: [https://tfhub.dev/deepmind/i3d-kinetics-400/1](https://tfhub.dev/deepmind/i3d-kinetics-400/1)
- Kinetics-400 Label Map: [https://github.com/deepmind/kinetics-i3d/blob/master/data/label_map.txt](https://github.com/deepmind/kinetics-i3d/blob/master/data/label_map.txt)