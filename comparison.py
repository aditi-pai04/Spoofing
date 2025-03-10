import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["LSTM", "Transformer", "Autoencoder", "CNN", "Improved CNN", "BiLSTM", "CNN-LSTM", "CNN-BiLSTM"]

# Corresponding accuracy values
accuracies = [49.72, 50.28, 50.40, 50.28, 50.28, 75.71, 83.99, 95.86]

# Number of epochs used for training each model
epochs = [50, 16, 16, 4, 4, 10, 6, 100]

# Set figure size
plt.figure(figsize=(10, 6))

# Create bar chart
bars = plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta', 'darkblue'])

# Annotate bars with accuracy and epoch values
for bar, acc, epoch in zip(bars, accuracies, epochs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f"{acc:.2f}%\n{epoch} epochs", ha='center', fontsize=10, fontweight='bold')

# Labels and title
plt.ylim(45, 100)
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.title("Comparative Accuracy of Different Models for GPS Spoofing Detection")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show plot
plt.xticks(rotation=20)  # Rotate model names for better readability
plt.tight_layout()
plt.show()
