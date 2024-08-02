
# Deep Learning Image Classification and Captioning

## Project Overview

This project implements deep learning techniques for image classification and image captioning using the COCO dataset. It aims to demonstrate the application of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) in computer vision tasks.

### What Was Done

- **Image Classification**: Implemented and trained CNN models for image classification.
- **Data Augmentation**: Applied various data augmentation techniques to improve model performance.
- **Hyperparameter Tuning**: Experimented with different learning rates and model architectures.
- **Transfer Learning**: Utilized pre-trained models and fine-tuned them for specific tasks.
- **Image Captioning**: Developed an RNN-based model for generating captions for images.
- **Model Evaluation**: Assessed model performance using accuracy metrics and loss curves.

### Key Findings

- CNNs outperformed MLPs in image classification tasks.
- Data augmentation and dropout helped in reducing overfitting.
- Transfer learning significantly improved model performance on the CIFAR10 dataset.
- The image captioning model showed promising results in generating relevant captions.

## Skills Demonstrated

- Deep Learning Architecture Design
- PyTorch Implementation
- Data Preprocessing and Augmentation
- Transfer Learning and Fine-tuning
- Hyperparameter Optimization
- Model Evaluation and Visualization

## Dataset

### Description

This project uses a subset of the COCO (Common Objects in Context) dataset for image captioning and a subset of TinyImageNet for classification tasks. Key details about the datasets include:

- **Full COCO Dataset:** Contains 330K images across 80 object categories. [Learn more about COCO](https://cocodataset.org/#home)
- **Our Subset (COCO_5070):** Consists of approximately 5070 images, each with five or more different descriptive captions.
- **TinyImageNet Subset:** Used for image classification with:
  - 30 different categories
  - 450 resized images (64x64 pixels) per category in the training set

## Visualizations

### 1. COCO Dataset Training - Vocabulary Size and Epochs Impact

<div align="center">
    <img width="375" alt="COCO Dataset Training - Vocabulary Size and Epochs Impact" src="https://github.com/user-attachments/assets/1c3ea052-1cd8-4625-b0c0-db64d3f4d1a9">
    <p><strong>Impact of Vocabulary Size and Epochs on Training and Validation Loss</strong></p>
</div>

#### Description:
This graph shows how varying vocabulary size affects training and validation loss over multiple epochs. The red line represents training loss, and the green line represents validation loss.

**Key Observations:**
- The training loss decreases consistently, indicating effective learning.
- Validation loss initially drops but starts to plateau, suggesting potential overfitting after the second epoch.
- A widening gap between training and validation loss highlights overfitting trends.

### 2. Learning Rate Comparison - Loss and Accuracy Curves

<div align="center">
    <img width="344" alt="Learning Rate Comparison - Loss and Accuracy Curves" src="https://github.com/user-attachments/assets/f7fad486-72bb-4c8c-a7c5-1a432457d312">
    <p><strong>Effect of Learning Rate on Loss Performance</strong></p>
</div>

<div align="center">
    <img width="329" alt="Learning Rate Comparison - Accuracy Curves" src="https://github.com/user-attachments/assets/a8397668-8ff7-4f56-9605-a5f4403b6a44">
    <p><strong>Effect of Learning Rate on Accuracy Performance</strong></p>
</div>

#### Description:
These graphs compare the performance of different learning rates over 30 epochs.

**Key Observations:**
- **Loss Curves:**
  - Learning rate 0.1 shows instability with high loss.
  - Learning rate 0.01 provides the best performance with the lowest loss.
  - Learning rate 0.001 demonstrates steady but slower improvement.

- **Accuracy Curves:**
  - Learning rate 0.01 achieves the highest accuracy during training.
  - Validation accuracy for 0.01 is slightly better than for 0.001.
  - Learning rate 0.1 results in poor accuracy, indicating it's too high.

### 3. Model Performance with Dropout

<div align="center">
    <img width="346" alt="Model Performance with Dropout" src="https://github.com/user-attachments/assets/14fa5bbd-42d3-4c24-a548-a0900e51f053">
    <p><strong>Training and Validation Performance with Dropout Regularization</strong></p>
</div>

#### Description:
These graphs illustrate the effect of dropout on training and validation loss and accuracy.

**Key Observations:**
- **Loss Curves:**
  - Training loss decreases consistently.
  - Validation loss shows fluctuations, suggesting some overfitting.

- **Accuracy Curves:**
  - Training accuracy improves steadily, reaching around 65%.
  - Validation accuracy reaches about 50%, indicating reasonable generalization but room for improvement.

### 4. Image Captioning Example

<div align="center">
    <img width="441" alt="Image Captioning Example" src="https://github.com/user-attachments/assets/5225d1f5-03fa-4e7a-85c0-ffa79d8ab96e">
    <p><strong>Generated Caption for an Example Image</strong></p>
</div>

#### Description:
This image demonstrates the captioning capability of the model.

**Generated Caption:** "A man riding a horse in an open field."

**Key Observations:**
- The caption effectively captures the main elements of the image: a person, a horse, and the field.
- Although the caption is accurate, some background details are missing.
- The generated caption is coherent and relevant to the image content.

## Conclusion

This project demonstrated the effectiveness of deep learning techniques in image classification and captioning tasks. Key conclusions include:

1. CNNs significantly outperform MLPs for image classification tasks.
2. Data augmentation and dropout are crucial for improving model generalization.
3. Transfer learning can dramatically improve performance, especially on smaller datasets.
4. RNN-based models show promise in generating coherent image captions.

These findings provide valuable insights into the application of deep learning in computer vision tasks and offer a foundation for further research and development in this field.

## How to Use

To explore and utilize this project:

1. **Clone the Repository**:
   ```
   git clone https://github.com/crishN144/DeepVision-Classification-Captioning
   cd DeepVision-Classification-Captioning
   ```

2. **Set Up the Environment**:
   - Create a virtual environment:
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows use `venv\Scripts\activate`
     ```
   - Install required dependencies:
     ```
     pip install -r requirements.txt
     ```

3. **Run the Notebook**:
   - Open Jupyter Notebook:
     ```
     jupyter notebook
     ```
   - Navigate to and open `Image_Classification_Captioning.ipynb`
   - Run the cells sequentially to reproduce the analysis and training

4. **Explore Models and Results**:
   - Experiment with different hyperparameters
   - Test the image captioning model on new images

5. **Modify and Extend**:
   - Feel free to modify the architectures, try different datasets, or implement new deep learning techniques

## Future Work

To further enhance this project, the following future works are proposed:

1. **Implement More Advanced Architectures**: Explore state-of-the-art models like Transformers for image captioning.
2. **Expand to Video Captioning**: Extend the captioning model to work with video data.
3. **Multi-modal Learning**: Integrate text and image data for more comprehensive understanding and generation tasks.
4. **Attention Mechanisms**: Implement attention mechanisms to improve the image captioning model's performance.
