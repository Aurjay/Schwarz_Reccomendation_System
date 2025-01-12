from flask import Flask, jsonify, request
import torch
import joblib
from torch.utils.data import DataLoader, Dataset
from Neural_net_model import Neural_Net
from flask_cors import CORS  # Import CORS
import os  # Import os for environment variable access

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Custom dataset class for inference (user, product_1, product_2)
class RecommenderDataset(Dataset):
    def __init__(self, user_ids, product_1_ids, product_2_ids):
        self.user_ids = user_ids
        self.product_1_ids = product_1_ids
        self.product_2_ids = product_2_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.product_1_ids[idx], self.product_2_ids[idx]

# Load encoders and model
user_encoder = joblib.load('./models/user_encoder.pkl')  
product_encoder = joblib.load('./models/product_encoder.pkl')

# Define the model architecture (same as during training)
n_users = len(user_encoder.classes_)
n_products = len(product_encoder.classes_)
device = torch.device("cpu")  # Ensure it defaults to CPU (as Cloud Run doesn't support GPU)

model = Neural_Net(
    n_users=n_users, 
    n_products=n_products, 
    n_factors=50, 
    embedding_dropout=0.1, 
    hidden=[100, 50], 
    dropouts=[0.4, 0.3]
).to(device)

# Load the best trained model (ensure the model architecture matches)
model.load_state_dict(torch.load('./models/best_model.pth', map_location=device))
model.eval()

# Default user ID (can be changed to any fixed user ID)
DEFAULT_USER_ID = 1

# Function to make product recommendations based on user and product pairs
def make_inference(model, user_id, product_1_id, product_2_id):
    # Prepare data for inference
    dataset = RecommenderDataset([user_id], [product_1_id], [product_2_id])
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_predictions = []
    with torch.no_grad():
        for user_ids, product_1_ids, product_2_ids in dataloader:
            user_ids = user_ids.to(device)
            product_1_ids = product_1_ids.to(device)
            product_2_ids = product_2_ids.to(device)

            # Forward pass to get predictions
            outputs = model(user_ids, product_1_ids, product_2_ids)
            _, predicted = torch.max(outputs, 1)

            # Decode the product IDs back to product names
            decoded_predictions = product_encoder.inverse_transform(predicted.cpu().numpy())
            all_predictions.extend(decoded_predictions)

    return all_predictions

# API route to get product recommendations
@app.route('/recommend', methods=['POST'])
def recommend_products():
    try:
        # Get the current product IDs from the request (user_id is fixed)
        data = request.get_json()
        current_product_ids = data.get('current_product_ids', [])

        if len(current_product_ids) < 2:
            return jsonify({"error": "At least two product IDs are required"}), 400

        # Make predictions for each consecutive product pair
        recommended_products = []
        for i in range(len(current_product_ids) - 1):
            product_1_id = current_product_ids[i]
            product_2_id = current_product_ids[i + 1]

            # Make predictions for the next product
            predicted_products = make_inference(model, DEFAULT_USER_ID, product_1_id, product_2_id)
            print(f"Predicted products for ({product_1_id}, {product_2_id}): {predicted_products}")

            recommended_products.extend(predicted_products)

        return jsonify({"recommended_products": recommended_products})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Get port from environment variable
    app.run(debug=False, host="0.0.0.0", port=port)  # Disable debug mode in production
