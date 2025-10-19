import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)

# Load California Housing dataset
print("Loading California Housing dataset...")
data = fetch_california_housing()
X, y = data.data, data.target

# Create DataFrame for better visualization
feature_names = data.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print("\nDataset info:")
print(df.describe())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

print(f"\nTraining set size: {X_train_tensor.shape[0]}")
print(f"Test set size: {X_test_tensor.shape[0]}")

# Create model
input_size = X_train_tensor.shape[1]
model = LinearRegressionModel(input_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
num_epochs = 1000
train_losses = []
val_losses = []

print("\nTraining started...")
for epoch in range(num_epochs):
    # Training
    model.train()
    train_pred = model(X_train_tensor)
    train_loss = criterion(train_pred, y_train_tensor)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor)
        val_loss = criterion(val_pred, y_test_tensor)
    
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Final evaluation
model.eval()
with torch.no_grad():
    train_pred = model(X_train_tensor).numpy()
    test_pred = model(X_test_tensor).numpy()

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\nFinal Results:")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Feature importance (coefficients)
weights = model.linear.weight.detach().numpy().flatten()
bias = model.linear.bias.detach().numpy()[0]

print(f"\nModel Parameters:")
print(f"Bias: {bias:.4f}")
print("\nFeature Weights (importance):")
for feature, weight in zip(feature_names, weights):
    print(f"{feature}: {weight:.4f}")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training and validation loss
axes[0, 0].plot(train_losses, label='Training Loss')
axes[0, 0].plot(val_losses, label='Validation Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Predictions vs Actual (Test set)
axes[0, 1].scatter(y_test, test_pred, alpha=0.6)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Values')
axes[0, 1].set_ylabel('Predicted Values')
axes[0, 1].set_title(f'Predictions vs Actual (Test Set)\nR² = {test_r2:.3f}')
axes[0, 1].grid(True)

# Plot 3: Feature importance
feature_importance = np.abs(weights)
sorted_idx = np.argsort(feature_importance)
axes[1, 0].barh(range(len(feature_importance)), feature_importance[sorted_idx])
axes[1, 0].set_yticks(range(len(feature_importance)))
axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[1, 0].set_xlabel('Absolute Weight')
axes[1, 0].set_title('Feature Importance (Absolute Weights)')
axes[1, 0].grid(True, axis='x')

# Plot 4: Residuals
residuals = y_test - test_pred.flatten()
axes[1, 1].scatter(test_pred, residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Residual Plot')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Display correlation matrix for top features
print(f"\nTop 3 most important features:")
top_features_idx = np.argsort(np.abs(weights))[-3:]
for idx in reversed(top_features_idx):
    print(f"{feature_names[idx]}: {weights[idx]:.4f}")

# Make predictions on a few sample points
print(f"\nSample Predictions:")
sample_indices = np.random.choice(len(X_test), 5, replace=False)
with torch.no_grad():
    for i, idx in enumerate(sample_indices):
        actual = y_test[idx]
        predicted = model(X_test_tensor[idx:idx+1]).item()
        print(f"Sample {i+1} - Actual: ${actual:.2f}00k, Predicted: ${predicted:.2f}00k")