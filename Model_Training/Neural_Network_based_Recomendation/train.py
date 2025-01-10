import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from Neural_net_model import Neural_Net

# Load dataset
df = pd.read_csv('../e_commerce_data_Cleaned.csv')

# Encode user_id and product_name
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['user_id'] = user_encoder.fit_transform(df['user_id'])
df['product_id'] = product_encoder.fit_transform(df['product_name'])

# Sort by order_id and user_id to maintain order of purchases
df = df.sort_values(['user_id', 'order_id', 'add_to_cart_order'])

# Generate sequences of products for each user within each order
product_sequences = []
for user_id, user_group in df.groupby('user_id'):
    products = user_group['product_id'].values
    for i in range(len(products) - 1):  # Create sequence of current and next product
        product_sequences.append((user_id, products[i], products[i + 1]))

# Convert sequences to a DataFrame
sequences_df = pd.DataFrame(product_sequences, columns=['user_id', 'current_product_id', 'next_product_id'])

# Split data into training, validation, and test sets
X_train, X_temp = train_test_split(sequences_df, test_size=0.2, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
train_data = torch.tensor(X_train[['user_id', 'current_product_id']].values, dtype=torch.long)
train_labels = torch.tensor(X_train['next_product_id'].values, dtype=torch.long)

val_data = torch.tensor(X_val[['user_id', 'current_product_id']].values, dtype=torch.long)
val_labels = torch.tensor(X_val['next_product_id'].values, dtype=torch.long)

test_data = torch.tensor(X_test[['user_id', 'current_product_id']].values, dtype=torch.long)
test_labels = torch.tensor(X_test['next_product_id'].values, dtype=torch.long)

# Define custom dataset
class RecommenderDataset(Dataset):
    def __init__(self, users, current_products, next_products):
        self.users = users
        self.current_products = current_products
        self.next_products = next_products

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # Returning both current and next product as inputs
        return self.users[idx], self.current_products[idx], self.next_products[idx]

# Create DataLoader for training, validation, and testing
train_dataset = RecommenderDataset(train_data[:, 0], train_data[:, 1], train_labels)
val_dataset = RecommenderDataset(val_data[:, 0], val_data[:, 1], val_labels)
test_dataset = RecommenderDataset(test_data[:, 0], test_data[:, 1], test_labels)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize the model
n_users = len(user_encoder.classes_)
n_products = len(product_encoder.classes_)

model = Neural_Net(n_users=n_users, n_products=n_products, n_factors=50)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Validation function
def validate(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for user_ids, current_product_ids, next_product_ids in val_loader:
            # Forward pass: Ensure the model receives both product_1 and product_2 as input
            outputs = model(user_ids, current_product_ids, next_product_ids)

            # Compute loss
            loss = criterion(outputs, next_product_ids)
            running_loss += loss.item()

            # Calculate the predicted product (with the highest score)
            _, predicted = torch.max(outputs, 1)
            
            # Calculate correct predictions
            correct_predictions += torch.sum(predicted == next_product_ids).item()
            total_predictions += len(next_product_ids)

    # Calculate accuracy and loss for the validation set
    val_loss = running_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for user_ids, current_product_ids, next_product_ids in train_loader:
        optimizer.zero_grad()

        # Forward pass: Ensure the model receives both product_1 and product_2 as input
        outputs = model(user_ids, current_product_ids, next_product_ids)  # Pass product_1 and product_2

        # Compute loss
        loss = criterion(outputs, next_product_ids)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    
    # Run validation after each epoch
    validate(model, val_loader, criterion)

# Save the model
torch.save(model.state_dict(), 'product_recommendation_model.pth')
torch.save(model, 'product_recommendation_model_full.pth')



# Test function
def test(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient calculation for testing
        for user_ids, current_product_ids, next_product_ids in test_loader:
            # Forward pass: Ensure the model receives both product_1 and product_2 as input
            outputs = model(user_ids, current_product_ids, next_product_ids)

            # Calculate the predicted product (with the highest score)
            _, predicted = torch.max(outputs, 1)
            
            # Check if the predicted product is in the actual products purchased by the user
            correct_predictions += torch.sum(predicted == next_product_ids).item()
            total_predictions += len(next_product_ids)

    # Calculate accuracy for the test set
    accuracy = correct_predictions / total_predictions

    print(f"Test Accuracy: {accuracy:.4f}")

# Run the test phase after training
test(model, test_loader)
