import matplotlib.pyplot as plt 
import json 
from pathlib import Path
import argparse

# ----------------------
# Parse command-line argument
# ----------------------
parser = argparse.ArgumentParser(description="Plot MNIST experiment metrics")
parser.add_argument("json_file", type=str, help="Path to the experiment JSON file")
args = parser.parse_args()

json_path = Path(args.json_file)

if not json_path.exists():
    print(f"Error: file {json_path} does not exist.")
    exit(1)
with open(json_path, "r") as f:
    data = json.load(f)

epochs = []
train_loss = []
test_accuracy = []

for epoch_key in sorted(data["metrics"].keys(), key=lambda x: int(x.split("_")[1])):
    epochs.append(int(epoch_key.split("_")[1]))
    train_loss.append(data["metrics"][epoch_key]["train_loss"])
    test_accuracy.append(data["metrics"][epoch_key]["test_accuracy"])

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color=color)
ax1.plot(epochs, train_loss, marker='o', color=color, label="Train Loss")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(epochs)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Test Accuracy (%)', color=color)
ax2.plot(epochs, test_accuracy, marker='x', color=color, label="Test Accuracy")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title(data["experiment_name"])
plt.show()
