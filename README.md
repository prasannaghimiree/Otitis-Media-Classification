Otitis Media Classification Project
====================================

This project is focused on classifying otitis media conditions using a deep learning model. The model leverages PyTorch and the Ignite library to perform efficient training and evaluation. The project also includes automated logging, model checkpointing, and data transformations to optimize the model's performance. It is designed for reproducibility and is ready for submission to Grand Challenge (https://grandchallenge.org/).

Table of Contents
-----------------
1. Overview
2. Features
3. Usage
4. Data Transformation Process
5. Model Architecture
6. Training and Evaluation
7. Results


Overview
--------
Otitis media is a common ear infection, particularly in children. Accurate diagnosis and classification of otitis media can significantly improve early intervention and treatment outcomes. This project builds a convolutional neural network (CNN) to classify otitis media from ear images. The code is implemented using PyTorch and the Ignite library, ensuring an efficient and scalable solution for medical image classification.

Features
--------
- **Data Handling**: Utilizes PyTorch `Dataset` and `DataLoader` classes for efficient image loading and batching.
- **Data Transformations**: Includes resizing, normalization, and augmentation techniques to improve generalization and model performance.
- **Model Architecture**: Built upon ResNet50, a proven deep learning model for image classification, adapted for otitis media classification with 8 categories.
- **Training and Evaluation**: Features built-in support for training, validation, and evaluation, along with real-time logging and metric tracking.
- **Checkpointing**: The best performing model is saved after every epoch based on accuracy, ensuring that the most optimal model is selected.
- **TensorBoard Logging**: Integrated with TensorBoard to visualize training loss, accuracy, and other metrics during model training.



Usage
------
1. **Dataset Preparation**:
- Place your image files and corresponding CSV files in the appropriate directories as specified in the code.
- Ensure the CSV file paths are correctly updated in the `ImageDataset` instantiation.

2. **Running the Model**:
- To start training, simply run the following command:
  ```
  python train.py
  ```
- The model will start training, and logs will be saved to the specified output directory. TensorBoard logs will also be saved for real-time visualization of the training progress.

3. **Evaluating the Model**:
- The model will be evaluated on the validation dataset after each epoch, and the results will be logged. The best model is automatically saved as a checkpoint.

Data Transformation Process
---------------------------
The data transformation process involves several steps to ensure the data is preprocessed correctly before being fed into the model. These steps include:
- **Resizing**: All input images are resized to a uniform size to match the model's input requirements (224x224).
- **Normalization**: Pixel values are normalized to a range of [0, 1] for better training stability.
- **Data Augmentation**: Various augmentations like random flipping and rotation are applied to increase the robustness of the model and prevent overfitting.

Model Architecture
------------------
The model is built upon the ResNet50 architecture, a well-known deep convolutional neural network that excels in image classification tasks. The following changes have been made to adapt it to otitis media classification:
- **Input Layer**: The first convolution layer is modified to accept images with 3 channels and a kernel size of 3, ensuring better feature extraction from medical images.
- **Output Layer**: The final fully connected layer has 8 units, corresponding to the 8 different classes of otitis media.

Training and Evaluation
-----------------------
The model is trained using the RMSProp optimizer with a learning rate of 0.005, which is suitable for medical image classification tasks. The loss function used is `CrossEntropyLoss`, which is ideal for multi-class classification problems.

During training, the model is evaluated after every epoch on the validation set to monitor its performance. The metrics (accuracy and loss) are calculated and logged, and the model checkpoints are saved when the accuracy improves.

The following key events are tracked during training:
- **Iteration Completed**: The training loss is logged after every 100 iterations.
- **Epoch Completed**: At the end of each epoch, the training and validation results are logged, including the accuracy and loss.

Results
-------
The model is trained for 5 epochs with the dataset and produces real-time logs on training and validation accuracy. The best model is saved based on the highest accuracy achieved during validation. The model is evaluated for both training and validation sets, and the results are visualized using TensorBoard.


For further assistance, feel free to open an issue or contribute to the project.

[Contact me on LinkedIn](https://www.linkedin.com/in/prasanna-ghimire-002335188/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
