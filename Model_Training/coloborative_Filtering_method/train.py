import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

# Load the cleaned dataset
file_path = "../e_commerce_data_Cleaned.csv"  
data = pd.read_csv(file_path)

# 1. Divide the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 2. Group by order_id to find products ordered together in the same order for training data
train_order_groups = train_data.groupby('order_id')['product_name'].apply(list)

# 3. Create a dictionary to store co-occurring products for each product in the training data
train_co_occurrence = defaultdict(set)

# Loop through each order in training data to count co-occurrences
for products in train_order_groups:
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            train_co_occurrence[products[i]].add(products[j])
            train_co_occurrence[products[j]].add(products[i])

# 4. Function to get top N product recommendations based on co-occurrence
def get_product_recommendations(product_name, co_occurrence, top_n=3):
    recommended_products = co_occurrence.get(product_name, [])
    return list(recommended_products)[:top_n]

def evaluate_recommendations(test_data, train_co_occurrence, top_n=3):
    y_true = []
    y_pred = []
    
    # For each order_id in the test data
    for order_id, group in test_data.groupby('order_id'):
        # Get the products ordered in the test set order
        ordered_products = group['product_name'].tolist()
        
        # For each ordered product, get the recommendations
        for product in ordered_products:
            # Get top N recommendations for the product
            recommendations = get_product_recommendations(product, train_co_occurrence, top_n)
            
            # Check if any of the recommendations match the ordered product
            for recommendation in recommendations:
                # Append the actual ordered product to y_true
                y_true.append(product)
                # Append the recommended product to y_pred
                y_pred.append(recommendation)
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return precision, recall, f1



# 6. Evaluate the model
precision, recall, f1 = evaluate_recommendations(test_data, train_co_occurrence, top_n=10)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 7. Save the trained co-occurrence model to a file for later inference
with open("train_co_occurrence_model.pkl", "wb") as file: 
    pickle.dump(train_co_occurrence, file)
    print("Model saved as 'train_co_occurrence_model.pkl'")
