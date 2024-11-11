from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Collect data
hidden_outputs = []
labels = []

# Randomly pick 1000 samples from each class
class_samples = {i: 0 for i in range(10)}
max_samples_per_class = 1000

for data, target in train_loader:
    data = data.view(-1, 28*28).numpy()
    target = target.numpy()
    
    # Forward pass up to the hidden layer
    out = input_layer.forward(data)
    out = relu1.forward(out)
    out = hidden_layer.forward(out)
    # No activation after hidden_layer since we want raw outputs
    
    for i in range(len(target)):
        label = target[i]
        if class_samples[label] < max_samples_per_class:
            hidden_outputs.append(out[i])
            labels.append(label)
            class_samples[label] += 1
        # Check if we have enough samples
        if all(count >= max_samples_per_class for count in class_samples.values()):
            break
    else:
        continue  # Only executed if the inner loop did NOT break
    break  # Inner loop was broken, so we break the outer loop as well


if __name__ == '__main__':
    hidden_outputs = np.array(hidden_outputs)
    labels = np.array(labels)

    # Apply T-SNE
    tsne = TSNE(n_components=2, random_state=42)
    hidden_tsne = tsne.fit_transform(hidden_outputs)


    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(hidden_tsne[:, 0], hidden_tsne[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=range(10))
    plt.title('T-SNE Visualization of Hidden Layer Outputs')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('feature_visualization.jpg')
    plt.show()
