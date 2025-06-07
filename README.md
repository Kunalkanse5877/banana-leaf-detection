🍌 Banana Leaf Disease Detection using Deep Learning

This project aims to detect diseases in banana leaves using deep learning techniques. It combines a custom Convolutional Neural Network (CNN) with attention and residual blocks, and also utilizes a ResNet50-based model for robust classification. The goal is to assist farmers and agriculturists in identifying leaf diseases early for effective crop management.

🚀 Features

- ✅ Custom CNN with residual & channel attention mechanisms
- ✅ Transfer Learning with ResNet50
- ✅ Preprocessing with Sobel edge detection & HSV conversion
- ✅ Model evaluation with accuracy, F1-score, confusion matrix
- ✅ Web interface for uploading banana leaf images and getting predictions

📁 Dataset

- **Source**: Manually curated + Kaggle-based banana leaf dataset
- **Classes**: Healthy, Sigatoka, Pestalotiopsis, Black Sigatoka, etc.
- **Structure**:
/dataset/
├── train/
├── test/
└── validation/

🧠 Models

- `best_advanced_model.keras`: Trained custom CNN model with attention
- `model_resnet_collab.keras`: Transfer learning with ResNet50 base

🧪 Evaluation Metrics

- ✅ Accuracy
- ✅ Precision, Recall, F1-score
- ✅ Confusion Matrix
- ✅ Loss vs Accuracy plots

💻 Web App

A simple Flask-based web interface (`app.py`) allows users to:
- Upload a banana leaf image
- View predicted disease name
- Display confidence score

🔧 Installation

1. Clone the repository:
 ```bash
 git clone https://github.com/Kunalkanse5877/banana-leaf-detection.git
 cd banana-leaf-detection
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

🌍 Deployment Options

You can deploy this project using:

* ✅ [Render](https://render.com/)
* ✅ Heroku
* ✅ Streamlit Cloud (if converted to Streamlit)

📦 File Structure

```
banana-leaf-detection/
├── app.py
├── Train.ipynb
├── Updated Code.ipynb
├── best_advanced_model.keras
├── model_resnet_collab.keras
├── static/
│   └── uploads/
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

🧰 Tech Stack

* Python, Keras, TensorFlow
* OpenCV, NumPy, Matplotlib
* Flask (for Web UI)

🙋‍♂️ Author

Kunal Kanse
GitHub: [@Kunalkanse5877](https://github.com/Kunalkanse5877)
Email: [kunalkanse92@gmail.com](mailto:kunalkanse92@gmail.com)

📌 License

This project is for educational and research purposes only.
