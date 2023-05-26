

```markdown
# Siamese Network for Face Recognition

This repository contains code for training and evaluating a Siamese Network for face recognition using TensorFlow. The network is trained on a dataset of anchor, positive, and negative images to learn embeddings that can distinguish between similar and dissimilar faces.
The Siamese model is inspired from the research paper titled "Siamese Neural Networks for One-shot Image Recognition" by Gregory Koch et al., which can be found [here](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) 

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python (version 3.6 or higher)
- TensorFlow (version 2.x)
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

## Getting Started

Follow the steps below to set up and run the project:

1. Clone the repository:
   ```shell
   git clone https://github.com/AbtgCoder/Real-Time-Facial-Recognition.git
   ```

2. Navigate to the repository folder:
   ```shell
   cd real-time-facial-recognition
   ```

3. Capture Images (Optional):
   - If you want to capture your own face images for training and testing, run the `image_capture.py` script.
   - Adjust the frame and region of interest as needed by modifying the code in `image_capture.py`.
   ```shell
   python image_capture.py
   ```
   
4. Prepare the Data:
   - Place your face images in the `data/anchor`, `data/positive`, and `data/negative` folders.
   - If you captured your own images using `image_capture.py`, they will be automatically saved in the respective folders.

5. Include the Prebuilt Model:
   - Download the prebuilt Siamese Network model file (siamese_model.h5) from     the following link:
   - [Download siamese_model.h5](https://drive.google.com/file/d/1BYxX6MZRZmhGFKG_vSt5x8-19elPZrIO/view?usp=sharing)
    Place the downloaded siamese_model.h5 file in the root folder of the repository.

6. Train the Siamese Network:
   - Run the `main.py` script to preprocess the images, train the model, and evaluate its performance on test data.
   ```shell
   python main.py
   ```

7. Real-Time Face Recognition:
   - Run the `real_time_face_rec.py` script to perform real-time face recognition using the trained model.
   ```shell
   python real_time_face_rec.py
   ```

## File Descriptions

- `main.py`: The main script to preprocess images, train the Siamese Network, and evaluate its performance.
- `data_preprocessing.py`: Contains functions for preprocessing the face images and preparing the training and testing datasets.
- `model_training.py`: Defines the Siamese Network architecture, trains the model, and saves the trained model.
- `model_evaluation.py`: Contains functions for evaluating the trained model using metrics like classification report, confusion matrix, and ROC curve.
- `real_time_face_rec.py`: Implements real-time face recognition using the trained model on live camera feed.
- `image_capture.py` (Optional): Allows capturing face images for custom training and testing data.


## Results and Evaluation

After training and evaluation, the model's performance will be displayed, including the classification report, confusion matrix, and ROC curve. The results will give insights into the model's ability to distinguish between similar and dissimilar faces.

## License

[MIT License](LICENSE)

The project is open source and released under the terms of the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, you can reach me at:
- Email:  [abtgofficial@gmail.com](mailto:abtgofficial@gmail.com)
```
