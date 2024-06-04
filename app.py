from flask import Flask, jsonify, request, send_file
import requests
from flask_cors import CORS

import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Set Matplotlib to use the 'Agg' backend
matplotlib.use('Agg')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def save_class_names(class_names):
    with open('class_names.txt', 'w') as file:
        for class_name in class_names:
            file.write(class_name + '\n')

def load_class_names():
    with open('class_names.txt', 'r') as file:
        saved_class_names = file.readlines()
    return [class_name.strip() for class_name in saved_class_names]

@app.route('/findfood', methods=['POST'])
def get_data():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img = cv2.imread(file_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png'))
    plt.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 101)
    )
    model.load_state_dict(torch.load('food101_resnet50.h5', map_location=device))
    model.to(device)
    model.eval()

    class_names = load_class_names()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def process_image(image_path):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)
        return image

    def predict(image_path, model, threshold=0.5):
        image = process_image(image_path)
        image = image.to(device)
        with torch.no_grad():
            output = model(image)
        probs, classes = output.softmax(dim=1).topk(1, dim=1)
        if probs.item() < threshold:
            return probs.item(), "Non-food item"
        else:
            return probs.item(), class_names[classes.item()]

    def display_prediction(image_path, model, threshold=0.5):
        prob, predicted_class = predict(image_path, model, threshold)
        image = Image.open(image_path).convert('RGB')
        plt.imshow(image)
        plt.title(f"Prediction: {predicted_class} ({prob * 100:.2f}%)")
        plt.axis('off')
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_image.png'))
        plt.close()
        return predicted_class

    food_name = display_prediction(file_path, model)
    print(food_name)

    url = f"http://localhost:5000/{food_name}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return jsonify(data)
    else:
        return jsonify({'error': 'Failed to retrieve data from external API'}), response.status_code



@app.route('/', methods=['GET'])
def get_data():
    return jsonify({'Name': 'SurajPandey'})



