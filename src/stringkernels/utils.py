import matplotlib.pyplot as plt
from collections import Counter

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

