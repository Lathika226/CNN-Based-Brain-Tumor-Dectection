# Brain Tumor Detection using Convolutional Neural Networks

A deep learning project for detecting brain tumors from medical images using Convolutional Neural Networks (CNNs).

## 📋 Project Overview

This project implements a CNN-based system for brain tumor detection from MRI or CT scan images. The system uses image processing techniques and deep learning to classify brain images as normal or containing tumors.

## 🚀 Features

- **Image Preprocessing**: Automatic image preprocessing including grayscale conversion, thresholding, and resizing
- **CNN Architecture**: Custom convolutional neural network for tumor detection
- **Segmentation**: Built-in tumor region segmentation using pre-trained models
- **GUI Interface**: User-friendly Tkinter-based graphical interface
- **Batch Processing**: Support for processing multiple images from directories
- **Real-time Results**: Live display of processing results and sample images

## 🛠️ Technologies Used

- **Python** - Main programming language
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **Tkinter** - GUI framework
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Scikit-learn** - Machine learning utilities

## 📁 Project Structure

```
brain-tumor-detection/
├── BrainTumor/
│   ├── BrainTumor.py          # Main GUI application
│   ├── test_load.py           # Model testing utilities
│   ├── diagnose_dataset.py    # Dataset diagnostic tool
│   ├── run.bat               # Windows batch file to run the app
│   ├── sample_dataset/       # Small sample dataset for testing
│   │   ├── no/              # Normal brain images (samples)
│   │   └── yes/             # Tumor images (samples)
│   ├── Model/               # Pre-trained models and weights
│   │   ├── model.json       # Model architecture
│   │   ├── model_weights.h5 # Model weights
│   │   ├── segmented_model.json
│   │   └── segmented_weights.h5
│   └── testImages/          # Test images directory
├── LoR/                     # Additional resources
└── README.md               # This file
```

## 📊 Dataset Information

**Note:** Due to GitHub's file size limits, the full training datasets are not included in this repository. The repository contains:

- ✅ **Pre-trained models** (ready to use)
- ✅ **Sample images** for testing
- ✅ **Complete source code**

### Getting the Full Datasets

To train the model or use larger datasets, you can:

1. **Use the sample dataset** included for basic testing
2. **Download brain MRI datasets** from public sources:
   - [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
   - [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
   - [Medical Segmentation Decathlon](http://medicaldecathlon.com/)

3. **Organize your dataset** as:
   ```
   your_dataset/
   ├── no/     # Normal brain images
   └── yes/    # Images with tumors
   ```

## 🖥️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install tensorflow opencv-python scikit-learn matplotlib numpy
   ```

4. **Run the application**
   ```bash
   python BrainTumor/BrainTumor.py
   ```
   Or use the batch file:
   ```bash
   BrainTumor\run.bat
   ```

## 📊 Usage

1. **Launch the Application**: Run `BrainTumor.py`
2. **Upload Dataset**: Click "Upload Dataset" and select a folder containing images
3. **Preprocess Data**: The system will automatically process and analyze your images
4. **View Results**: See processing statistics and sample images
5. **Train Model**: Use the training functionality (if available)
6. **Test Images**: Load individual images for tumor detection

## 🎯 Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## 📈 Model Performance

- **Architecture**: Custom CNN with convolutional and pooling layers
- **Input Size**: 64x64 grayscale images
- **Output**: Binary classification (Normal/Tumor)
- **Preprocessing**: Otsu thresholding and morphological operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset sources and preprocessing techniques
- OpenCV and TensorFlow communities
- Medical imaging research community


---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Not intended for clinical diagnosis. Always consult with medical professionals for actual diagnosis.
