import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from Neural_net_model import Neural_Net
import joblib 

# Load dataset
df = pd.read_csv('../e_commerce_data_Cleaned.csv')

# Encode user_id and product_name
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

# Fit the encoders to user and product data
df['user_id'] = user_encoder.fit_transform(df['user_id'])
df['product_id'] = product_encoder.fit_transform(df['product_name'])

# Sort by order_id and user_id to maintain order of purchases
df = df.sort_values(['user_id', 'order_id', 'add_to_cart_order'])

# Generate product triplets for each user where product[i] and product[i+1] have the next product[i+2] in the same order
product_sequences = []
for user_id, user_group in df.groupby('user_id'):
    products = user_group['product_id'].values
    for i in range(len(products) - 2):  
        product_sequences.append((user_id, products[i], products[i + 1], products[i + 2]))  

# Convert sequences to a DataFrame
sequences_df = pd.DataFrame(product_sequences, columns=['user_id', 'product_1_id', 'product_2_id', 'next_product_id'])

# Convert to PyTorch tensors
data = torch.tensor(sequences_df[['user_id', 'product_1_id', 'product_2_id']].values, dtype=torch.long)
labels = torch.tensor(sequences_df['next_product_id'].values, dtype=torch.long)

# Define custom dataset
class RecommenderDataset(Dataset):
    def __init__(self, users, product_1, product_2, next_product):
        self.users = users
        self.product_1 = product_1
        self.product_2 = product_2
        self.next_product = next_product

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.product_1[idx], self.product_2[idx], self.next_product[idx]

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    print(f"Training fold {fold + 1}")

    # Split into training and validation datasets
    train_data, val_data = data[train_idx], data[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    train_dataset = RecommenderDataset(train_data[:, 0], train_data[:, 1], train_data[:, 2], train_labels)
    val_dataset = RecommenderDataset(val_data[:, 0], val_data[:, 1], val_data[:, 2], val_labels)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize the model
    n_users = len(user_encoder.classes_)
    n_products = len(product_encoder.classes_)

    model = Neural_Net(
        n_users=n_users, 
        n_products=n_products, 
        n_factors=50, 
        embedding_dropout=0.1,  # For dropout after embeddings
        hidden=[100, 50],       # Hidden layer sizes
        dropouts=[0.4, 0.3]     # Dropout rates for hidden layers
    ).to(device)

    # Define loss and optimizer (L2 regularization with weight decay)
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization (weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  

    # Validation function
    def validate(model, val_loader, criterion):
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        predictions = []
        with torch.no_grad():
            for user_ids, product_1_ids, product_2_ids, next_product_ids in val_loader:
                user_ids, product_1_ids, product_2_ids, next_product_ids = user_ids.to(device), product_1_ids.to(device), product_2_ids.to(device), next_product_ids.to(device)
                outputs = model(user_ids, product_1_ids, product_2_ids)
                loss = criterion(outputs, next_product_ids)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += torch.sum(predicted == next_product_ids).item()
                total_predictions += len(next_product_ids)

                # Collect predictions to print later
                predictions.append((user_ids[:5], product_1_ids[:5], product_2_ids[:5], next_product_ids[:5], predicted[:5]))

        val_loss = running_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        return val_loss, accuracy, predictions

    # Training loop with early stopping
    epochs = 5
    patience = 5
    best_val_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for user_ids, product_1_ids, product_2_ids, next_product_ids in train_loader:
            user_ids, product_1_ids, product_2_ids, next_product_ids = user_ids.to(device), product_1_ids.to(device), product_2_ids.to(device), next_product_ids.to(device)

            optimizer.zero_grad()
            outputs = model(user_ids, product_1_ids, product_2_ids)
            loss = criterion(outputs, next_product_ids)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        val_loss, val_accuracy, val_predictions = validate(model, val_loader, criterion)
        scheduler.step()  
        
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Training Loss: {running_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), f'output_models/best_model_fold{fold + 1}.pth')  
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping triggered.")
                break

    # Save encoders after each fold (you can save them only once if preferred)
    joblib.dump(user_encoder, f'output_models/user_encoder_fold{fold + 1}.pkl')
    joblib.dump(product_encoder, f'output_models/product_encoder_fold{fold + 1}.pkl')

# Test function (after cross-validation and training)
def test(model, test_loader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for user_ids, product_1_ids, product_2_ids, next_product_ids in test_loader:
            user_ids, product_1_ids, product_2_ids, next_product_ids = user_ids.to(device), product_1_ids.to(device), product_2_ids.to(device), next_product_ids.to(device)
            outputs = model(user_ids, product_1_ids, product_2_ids)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += torch.sum(predicted == next_product_ids).item()
            total_predictions += len(next_product_ids)

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy: {accuracy:.4f}")

# Load the best model from the last fold and test
best_model = Neural_Net(n_users=n_users, n_products=n_products, n_factors=50, embedding_dropout=0.3).to(device)
best_model.load_state_dict(torch.load(f'output_models/best_model_fold{fold + 1}.pth'))
test(best_model, val_loader)  
