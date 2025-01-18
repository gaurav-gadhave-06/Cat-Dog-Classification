# Cat vs Dog Image Classification

## Project Overview
This project is a binary classification problem that uses convolutional neural networks (CNNs) to distinguish between images of cats and dogs. The notebook utilizes transfer learning with the EfficientNetB0 model for improved accuracy and faster training.

## Dependencies
Ensure you have the following Python libraries installed before running the notebook:

```bash
pip install tensorflow matplotlib numpy pandas scikit-learn seaborn
```

### Key Libraries Used
- `TensorFlow` and `Keras`: For building, training, and evaluating the deep learning model.
- `matplotlib` and `seaborn`: For data visualization and plotting results.
- `Pandas` and `NumPy`: For data manipulation and preprocessing.
- `sklearn`: For model evaluation metrics such as confusion matrix and classification report.

## Dataset
- **Source**: The dataset consists of cat and dog images stored in a zip file.
- **Preprocessing**:
  - The images are extracted and stored in the `/content/cat_dog` directory.
  - Data augmentation is applied using the `ImageDataGenerator` class to prevent overfitting.
  - Training and validation datasets are split and resized to \(150 \times 150 \times 3\) dimensions.

## Model Architecture
The project uses the EfficientNetB0 model as a base for transfer learning:

1. **EfficientNetB0 Base**:
   - Pre-trained on the ImageNet dataset.
   - The top layers are excluded (`include_top=False`).

2. **Custom Layers**:
   - Global Average Pooling layer to reduce feature dimensions.
   - Dense layer with 256 units and ReLU activation.
   - Dropout layer (0.5) to prevent overfitting.
   - Output layer with a sigmoid activation for binary classification.

### Model Compilation
- Optimizer: Adam with a learning rate of 0.0001.
- Loss Function: Binary Cross-Entropy.
- Metrics: Accuracy.

## Training and Evaluation
- **Training**:
  - Augmented data is passed through the model in batches.
  - Early stopping is used to monitor validation loss and avoid overfitting.

- **Evaluation**:
  - Confusion matrix and classification report for detailed metrics.
  - ROC curve and AUC for performance visualization.

## Visualization
The notebook includes visualizations to:
- Display sample images from the dataset.
- Plot training and validation accuracy/loss over epochs.
- Show the confusion matrix and classification metrics.

## Usage
### Running the Notebook
1. Extract the dataset into the `/content` directory.
2. Open the notebook in your preferred IDE (e.g., Google Colab, Jupyter).
3. Run all cells sequentially to:
   - Preprocess the data.
   - Build and train the model.
   - Evaluate the model and visualize results.

### Example Commands
```python
# Extract dataset
import zipfile
zip_ref = zipfile.ZipFile('/content/cat-dog-images-for-classification.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

# Plot images from the generator
plot_images_from_generator(train_generator, num_images=4)
```

## Results
The trained model achieves:
- High accuracy in distinguishing between cats and dogs.
- ROC-AUC score indicating strong classification performance.

## Acknowledgements
- The EfficientNetB0 pre-trained model is sourced from TensorFlow/Keras.
- Data augmentation and preprocessing are performed using the TensorFlow `ImageDataGenerator` class.
