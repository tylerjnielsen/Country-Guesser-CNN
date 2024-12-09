from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import torch.nn as nn
from torchvision import models
import pycountry

app = Flask(__name__)

# class for loading the model
class MultiInputModel(nn.Module):
    def __init__(self, num_countries, num_cities, additional_features_dim):
        super().__init__()
        self.resnet = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.additional_fc = nn.Sequential(
            nn.Linear(additional_features_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2163227635),
            nn.Linear(512, 64)
        )

        self.final_fc = nn.Linear(in_features + 64, num_countries + num_cities)

    def forward(self, image, additional_features):
        img_features = self.resnet(image)
        additional_features = self.additional_fc(additional_features)
        combined_features = torch.cat((img_features, additional_features), dim=1)
        return self.final_fc(combined_features)

# loading the data and country/city mapping
country_to_index, city_to_index = {}, {}
index_to_country, index_to_city = {}, {}
with open('country_mapping.txt', 'r') as f:
    for line in f:
        country, index = line.strip().split(':')
        country_to_index[country] = int(index)
        index_to_country[int(index)] = country

with open('city_mapping.txt', 'r') as f:
    for line in f:
        city, index = line.strip().split(':')
        city_to_index[city] = int(index)
        index_to_city[int(index)] = city

ground_truth = pd.read_csv('test.csv')
ground_truth.reset_index(drop=True, inplace=True)

def country_name_to_code(country_name):
    try:
        return pycountry.countries.get(name=country_name).alpha_2.upper()
    except AttributeError:
        return None

def country_code_to_name(country_code):
    try:
        return pycountry.countries.get(alpha_2=country_code).name
    except AttributeError:
        return None

categorical_columns = ['region', 'sub-region', 'drive_side', 'climate', 'soil', 'land_cover']

DATASET_FOLDER = 'Newdataset'

# transforming images to ensure that they
# are in the correct format for our model to
# make predictions
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# loading the model from the .pth file
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiInputModel(len(country_to_index), len(city_to_index), len(categorical_columns)).to(device)
model.load_state_dict(torch.load('90country70city.pth', map_location=device))
model.eval()

score = 0

@app.route("/", methods=["GET", "POST"])
def index():
    global score

    # just extracting info from the mapping
    countries = [country_code_to_name(code) for code in country_to_index.keys()]
    cities = list(city_to_index.keys())

    if request.method == "POST":
        image_id = int(request.form.get("image_id"))

        # obtaining the correct answers
        true_row_index = image_id + 0  # the +0 is there because there was some offset issues but they got resolved. leaving the 0 there cuz why not
        true_country_code = ground_truth.iloc[true_row_index]['country']
        true_city = ground_truth.iloc[true_row_index]['city']
        true_country_name = country_code_to_name(true_country_code)

        # processing image before sending to model
        img_path = os.path.join(DATASET_FOLDER, f"{image_id}.png")
        if not os.path.exists(img_path):
            return render_template("index.html", error="Image file not found.", score=score, countries=countries, cities=cities)

        image = Image.open(img_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)

        additional_features = torch.zeros((1, len(categorical_columns)), device=device)

        # actually making the predictions
        with torch.no_grad():
            outputs = model(input_image, additional_features)
            country_pred = torch.argmax(outputs[0, :len(country_to_index)]).item()
            city_pred = torch.argmax(outputs[0, len(country_to_index):]).item()

        predicted_country_code = index_to_country.get(country_pred, "Unknown")
        predicted_country_name = country_code_to_name(predicted_country_code)
        predicted_city = index_to_city.get(city_pred, "Unknown")

        return jsonify({
            "image_id": image_id,
            "true_country": true_country_name,
            "true_city": true_city,
            "predicted_country": predicted_country_name,
            "predicted_city": predicted_city,
        })

    return render_template("index.html", score=score, countries=countries, cities=cities)

@app.route('/Newdataset/<filename>')
def serve_image(filename):
    return send_from_directory(DATASET_FOLDER, filename)


@app.route("/predict", methods=["POST"])
def predict():
    # this function is for predictions on our dataset, it is quite similar to the
    # function below, but that one is for user uploaded images
    try:
        image_id = request.form.get("image_id")
        country = request.form.get("country")
        city = request.form.get("city")

        app.logger.info(f"Received: image_id={image_id}, country={country}, city={city}")

        img_path = os.path.join(DATASET_FOLDER, f"{image_id}.png")
        if not os.path.exists(img_path):
            return jsonify({"error": "Image not found."}), 404

        image = Image.open(img_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)

        additional_features = torch.zeros((1, len(categorical_columns)), device=device)

        with torch.no_grad():
            outputs = model(input_image, additional_features)
            country_pred = torch.argmax(outputs[0, :len(country_to_index)]).item()
            city_pred = torch.argmax(outputs[0, len(country_to_index):]).item()

        predicted_country_code = index_to_country.get(country_pred, "Unknown")
        predicted_country_name = country_code_to_name(predicted_country_code)
        predicted_city = index_to_city.get(city_pred, "Unknown")

        return jsonify({
            "true_country": country_code_to_name(ground_truth.iloc[int(image_id)]['country']),
            "true_city": ground_truth.iloc[int(image_id)]['city'],
            "predicted_country": predicted_country_name,
            "predicted_city": predicted_city,
        })
    except Exception as e:
        app.logger.error(f"Error: {e}")  # Debugging output
        return jsonify({"error": str(e)}), 500


@app.route('/upload-predict', methods=['POST'])
def upload_predict():
    # prediction function for user uploaded images
    # much of the logic is the same as before
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)

        additional_features = torch.zeros((1, len(categorical_columns)), device=device)

        with torch.no_grad():
            outputs = model(input_image, additional_features)
            country_pred = torch.argmax(outputs[0, :len(country_to_index)]).item()
            city_pred = torch.argmax(outputs[0, len(country_to_index):]).item()

        predicted_country_code = index_to_country.get(country_pred, "Unknown")
        predicted_country_name = country_code_to_name(predicted_country_code)
        predicted_city = index_to_city.get(city_pred, "Unknown")

        return jsonify({
            "predicted_country": predicted_country_name,
            "predicted_city": predicted_city
        })
    except Exception as e:
        app.logger.error(f"Error in upload_predict: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
