import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from aqi_predictor.training_pipeline.process import (
    train_scaled,
    test_scaled,
    train_labels,
    test_labels,
    feature_columns,
)
from sklearn.metrics import accuracy_score, mean_squared_error 
import shap

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step
        return out, hn, cn




trainX = torch.tensor(train_scaled, dtype=torch.float32)
trainY = torch.tensor(train_labels, dtype=torch.float32)
testX = torch.tensor(test_scaled, dtype=torch.float32)
testY = torch.tensor(test_labels, dtype=torch.float32)

# Ensure targets are column vectors to match model output shape (n_samples, 1)
if trainY.dim() == 1:
    trainY = trainY.unsqueeze(1)
if testY.dim() == 1:
    testY = testY.unsqueeze(1)

# Instantiate the model with the correct input dimension (number of features)
n_features = trainX.shape[2]
model = LSTMModel(input_dim=n_features, hidden_dim=100, layer_dim=1, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
h0, c0 = None, None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, h0, c0 = model(trainX, h0, c0)

    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    h0, c0 = h0.detach(), c0.detach()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    predicted, _, _ = model(testX, None, None)

predicted = predicted.squeeze().cpu().numpy()
original = testY.squeeze().cpu().numpy()

mse = np.mean((predicted - original) ** 2)
print(f'Test MSE: {mse:.4f}')

accuracy = 1 - (np.mean(np.abs(predicted - original)) / np.mean(np.abs(original)))
print(f'Test Accuracy: {accuracy:.4f}')

plt.figure(figsize=(12, 6))
plt.plot(original, label='Original Data')
plt.plot(predicted, label='Predicted Data', linestyle='--')
plt.title('LSTM Model Predictions vs. Original Data')
plt.xlabel('Sample')
plt.ylabel('Scaled us_aqi')
plt.legend()
plt.show()



shap.initjs()

# SHAP: use a model-agnostic explainer for PyTorch LSTM (TreeExplainer does not support
# PyTorch modules). KernelExplainer works for any model but is slower.

# Create a prediction wrapper that accepts flattened numpy arrays and reshapes
def model_predict_flat(x_flat):
    # Accept both 1D (single sample) and 2D inputs
    x_flat = np.asarray(x_flat)
    if x_flat.ndim == 1:
        x_flat = x_flat.reshape(1, -1)
    # x_flat shape now: (n_samples, seq_len * features)
    arr = x_flat.reshape((x_flat.shape[0], trainX.shape[1], trainX.shape[2]))
    t = torch.tensor(arr, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out, _, _ = model(t)
    # Ensure a 1D numpy array is always returned (n_samples,)
    result = out.squeeze().cpu().numpy()
    return np.atleast_1d(result)

background = trainX.numpy().reshape(trainX.shape[0], -1)
if background.shape[0] > 50:
    background = background[:50]

explainer = shap.KernelExplainer(model_predict_flat, background)

testX_flat = testX.numpy().reshape(testX.shape[0], -1)
# nsamples controls runtime; increase for more accurate estimates
shap_values = explainer.shap_values(testX_flat, nsamples=100)

print("Variable Importance Plot - Global Interpretation")
plt.figure(figsize=(12, 6))
# Feature names per time-step (flattened sequence features).
# Build names matching the flattened ordering used for KernelExplainer.
seq_len = trainX.shape[1]
n_features = trainX.shape[2]
# Determine base feature names (fall back to generic names if mismatch)
if len(feature_columns) == n_features:
    base_names = feature_columns
else:
    base_names = [f"feat{i}" for i in range(n_features)]

# Flattened ordering: for each time step (oldest->newest), list all features.
feature_names = []
for t in range(seq_len):
    for f in range(n_features):
        lag = seq_len - t
        feature_names.append(f"{base_names[f]}_t-{lag}")

shap.summary_plot(shap_values, testX_flat, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot for LSTM Model")
plt.show()
