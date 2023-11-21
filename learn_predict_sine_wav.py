
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network model
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = self.fc2(x)
        return x

# Generate training data
num_samples = 1000
X = np.linspace(0, 2 * np.pi, num_samples).reshape(-1, 1)
Y = np.sin(X)
X_tensor = torch.Tensor(X).float()
Y_tensor = torch.Tensor(Y).float()

# Create the model, loss function, and optimizer
model = SineNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()

# Test the model
with torch.no_grad():
    test_x = torch.Tensor(np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)).float()
    predicted = model(test_x)

# Plot the results
plt.plot(test_x, predicted, label='Predicted')
plt.plot(X, Y, label='True')
plt.legend()
plt.title("Predicted vs True Sine Wave")
plt.show()
