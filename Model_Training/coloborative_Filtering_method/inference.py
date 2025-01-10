import pickle

# Load the trained co-occurrence model from the saved file
with open("train_co_occurrence_model.pkl", "rb") as file:
    train_co_occurrence = pickle.load(file)

# Function to get top N product recommendations based on co-occurrence
def get_product_recommendations(product_name, co_occurrence, top_n=3):
    recommended_products = co_occurrence.get(product_name, [])
    return list(recommended_products)[:top_n]

# Example usage - you can replace the input with any product name
def recommend_products_for_new_user(new_product, top_n=3):
    recommendations = get_product_recommendations(new_product, train_co_occurrence, top_n)
    print(f"Recommended products for '{new_product}':")
    for product in recommendations:
        print(f"- {product}")

# Example: Recommend products for a new user who ordered 'Apple'
if __name__ == "__main__":
    new_product = "fresh fruits"
    top_n = 3
    recommend_products_for_new_user(new_product, top_n)
