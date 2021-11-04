from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pkg_resources

def load_sample_data():
    
    data_path = pkg_resources.resource_filename('stringkernels', 'data/')
    data_names = [
        "samples_train",
        "samples_validation",
        "ancestry_train",
        "ancestry_validation",
        "reference",
        "populations"
    ]
    out = { data_name: np.load(os.path.join(data_path, data_name+".npy")) for data_name in data_names }

    return out

def plot_label_distribution(labels):
    Counts = Counter(labels)
    labels = Counts.keys()
    n_labels = Counts.values()
    perc = [n/len(labels) for n in n_labels]
    plt.pie(perc, labels=labels)
    plt.show()

def plot_accuracies(accuracy_dictionary):
    plt.bar(accuracy_dictionary.keys(), accuracy_dictionary.values())
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.show()

