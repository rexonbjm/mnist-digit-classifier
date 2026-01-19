# MNIST Digit Classifier - Docker Deployment

An interactive web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network trained on the MNIST dataset. Draw digits on the canvas and get real-time predictions!
Live Demo: https://demo.reksonbajimaya.com.np/mnist

## ✨ Features

- 🎨 **Interactive Canvas**: Draw digits with mouse or touch
- 🤖 **Real-time Predictions**: Instant AI predictions as you draw
- 📊 **Confidence Visualization**: See probability distribution for all digits
- 🐳 **Docker Ready**: Easy deployment with Docker
- ⚡ **FastAPI Backend**: High-performance async API
- 🎯 **CNN Model**: Custom TensorFlow/Keras convolutional neural network

## 🚀 Quick Start

### Prerequisites

- Docker installed on your system
- Or Python 3.10+ (for local development)

### Option 1: Docker Deployment (Recommended)

1. **Build the Docker image**:
   ```bash
   docker build -t mnist-classifier .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 mnist-classifier
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

### Option 2: Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Open your browser**:
   ```
   http://localhost:8000
   ```

## 📁 Project Structure

```
mnist-digit-classifier/
├── app/                  # Application directory
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── predict_image.py # CLI prediction script
│   └── static/          # Frontend files
│       ├── index.html   # Main web page
│       ├── style.css    # Styling
│       └── script.js    # Canvas & API interaction
├── model/
│   └── mnist_model.keras # Trained model
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md          # This file
```

## 🎯 API Endpoints

### `GET /`
Serves the main web application

### `POST /predict`
Predicts a digit from a base64-encoded image

**Request Body**:
```json
{
  "image": "data:image/png;base64,..."
}
```

**Response**:
```json
{
  "digit": 7,
  "confidence": 0.9856,
  "probabilities": {
    "0": 0.0001,
    "1": 0.0002,
    ...
    "7": 0.9856,
    ...
  }
}
```

### `GET /health`
Health check endpoint

## 🔧 Model Architecture

- **Input**: 28x28 grayscale images
- **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling**: 2x2 pool size
- **Flatten Layer**
- **Dense Layer**: 128 neurons, ReLU activation
- **Output Layer**: 10 neurons (digits 0-9), Softmax activation

**Training**: 5 epochs on 60,000 MNIST training images  
**Test Accuracy**: ~98%

## 🐳 Docker Commands

**Build image**:
```bash
docker build -t mnist-classifier .
```

**Run container**:
```bash
docker run -p 8000:8000 mnist-classifier
```

**Run in detached mode**:
```bash
docker run -d -p 8000:8000 --name mnist-app mnist-classifier
```

**View logs**:
```bash
docker logs mnist-app
```

**Stop container**:
```bash
docker stop mnist-app
```

**Remove container**:
```bash
docker rm mnist-app
```

## 🎨 Usage Tips

1. **Draw clearly**: Use smooth, connected strokes
2. **Center your digit**: Keep it in the middle of the canvas
3. **Size matters**: Make your digit reasonably large
4. **Clear between digits**: Click "Clear Canvas" for fresh predictions
5. **Watch the bars**: The confidence distribution shows how certain the model is

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: Pillow, NumPy
- **Frontend**: Vanilla JavaScript, HTML5 Canvas, CSS3
- **Deployment**: Docker

## 📝 License

MIT License - Feel free to use this project for learning and development!

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 🙏 Acknowledgments

- MNIST dataset from Yann LeCun's website
- TensorFlow/Keras team for the amazing framework
- FastAPI for the modern web framework
