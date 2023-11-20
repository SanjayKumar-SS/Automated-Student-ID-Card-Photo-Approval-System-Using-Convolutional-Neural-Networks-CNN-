from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import pandas as pd

app = Flask(__name__)
app.secret_key = 'ASDFGHJKL*123456789'

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 32 * 32, 128)
        self.relu3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
model = SimpleCNN()

model = torch.load("./Photo_ClassifierV1.pt")

def predict_img(image_path):
    # Load the model into evaluation mode
    model.eval()
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    # Check if the image has 4 channels (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    preprocessed_image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(preprocessed_image)

    predicted_class = 1 if output <= 0.5 else 0  # Assuming 0.5 as the threshold

    return predicted_class

# Initialize DataFrame to store data
columns = ['Name', 'Date of Birth', 'Department', 'Register Number', 'Batch', 'Photo Status',"Photo Location"]
data_df = pd.DataFrame(columns=columns)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        name = request.form['name']
        dob = request.form['dob']
        department = request.form['department']
        reg_number = request.form['regNumber']
        batch = request.form['batch']
        photo = request.files['photo']

        # Save uploaded photo
        photo_path = f"uploads/{secure_filename(photo.filename)}"
        photo.save(photo_path)

        # Preprocess the image
        # Make predictions using your model
        # For simplicity, let's assume the model returns 0 or 1
        prediction = predict_img(photo_path) # Replace this with your actual model prediction logic
        print(prediction)

        # Save data to Excel file
        data_df.loc[len(data_df)] = [name, dob, department, reg_number, batch, prediction,photo_path]
        
        excel_file_path = 'document_verification_data.xlsx'

        # Check if the file exists
        if os.path.isfile(excel_file_path):
            # Load the existing data from the Excel file
            existing_data = pd.read_excel(excel_file_path, sheet_name='Sheet1')

            # Append the new data
            updated_data = pd.concat([existing_data, data_df], ignore_index=True)
            updated_data.to_excel(excel_file_path, index=False, sheet_name='Sheet1')

        else:
            # If the file doesn't exist, create a new one
            data_df.to_excel(excel_file_path, index=False, sheet_name='Sheet1')

        # Return the prediction result as JSON
        return jsonify({'result': prediction})

    except Exception as e:
        # Getting error ImportError: Missing optional dependency 'openpyxl'.  Use pip or conda to install openpyxl. even though it is not used
        print(f"Error: {str(e)}")
        return jsonify({'result': 0})

@app.route('/submit', methods=['POST'])
def submit():
    data_df.to_excel('document_verification_data.xlsx', index=False)

    return jsonify({'message': 'Data submitted successfully!'})

if __name__ == '__main__':
    app.run(debug=True)