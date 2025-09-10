# Brain Tumor Detection

This project uses deep learning to classify brain MRI images into four categories: **glioma**, **meningioma**, **no_tumor**, and **pituitary**. It includes a Flask web app for uploading MRI images and viewing predictions.

---

## Features

- Train a Convolutional Neural Network (CNN) on MRI images
- Web interface for image upload and tumor type prediction
- Clean project structure with empty folders for data and models

---

## Folder Structure

```
brain_tumor_detection/
│
├── app.py                # Flask web application
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── .gitignore
├── README.md
│
├── dataset/              # Place your full dataset here (not included in repo)
│   └── .gitkeep
├── training/             # Place your training subset here (not included in repo)
│   └── .gitkeep
├── uploads/              # Uploaded images (used by web app)
│   └── .gitkeep
├── models/               # Trained models will be saved here
│   └── .gitkeep
│
└── templates/
    ├── index.html        # Main upload page
    └── result.html       # Prediction result page
```

---

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/DarkFlame2205/brain_tumour_detection.git
cd brain_tumour_detection
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Download the Dataset

- **Dataset is NOT included in this repository.**
- Download the MRI dataset from your preferred source (e.g., [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) or another link).
- Place the images in the `dataset/` and `training/` folders as required by `train_model.py`.

### 4. Train the Model

```sh
python train_model.py
```
- The trained model will be saved in the `models/` folder.

### 5. Run the Web App

```sh
python app.py
```
- Open your browser and go to `http://127.0.0.1:5000/` to use the app.

---

## Notes

- The folders `dataset/`, `training/`, `uploads/`, and `models/` are included as empty folders with `.gitkeep` files. Add your data as needed.
- Do **not** upload large datasets or model files to the repository.

---

## License

This project is for educational purposes. Please check dataset sources for
