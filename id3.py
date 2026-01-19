import math
from collections import Counter

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent


def information_gain(data, labels, attribute_index):
    base_entropy = entropy(labels)
    total = len(labels)

    subsets = {}
    for row, label in zip(data, labels):
        value = row[attribute_index]
        subsets.setdefault(value, []).append(label)

    weighted_entropy = 0.0
    for subset_labels in subsets.values():
        weighted_entropy += (len(subset_labels) / total) * entropy(subset_labels)

    return base_entropy - weighted_entropy


def majority_label(labels):
    return Counter(labels).most_common(1)[0][0]


def id3(data, labels, attributes):
    # If all labels are the same
    if len(set(labels)) == 1:
        return labels[0]

    # If no attributes left
    if not attributes:
        return majority_label(labels)

    # Choose best attribute
    gains = [information_gain(data, labels, attr) for attr in attributes]
    best_attr = attributes[gains.index(max(gains))]

    tree = {best_attr: {}}

    values = set(row[best_attr] for row in data)

    for value in values:
        sub_data = []
        sub_labels = []

        for row, label in zip(data, labels):
            if row[best_attr] == value:
                sub_data.append(row)
                sub_labels.append(label)

        remaining_attrs = [a for a in attributes if a != best_attr]

        if not sub_data:
            tree[best_attr][value] = majority_label(labels)
        else:
            tree[best_attr][value] = id3(
                sub_data,
                sub_labels,
                remaining_attrs
            )

    return tree


def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    attr = next(iter(tree))
    value = sample[attr]

    if value in tree[attr]:
        return predict(tree[attr][value], sample)
    else:
        return None


if __name__ == "__main__":
    # Play Tennis dataset
    X = [
        ['Sunny', 'Hot', 'High', 'Weak'],
        ['Sunny', 'Hot', 'High', 'Strong'],
        ['Overcast', 'Hot', 'High', 'Weak'],
        ['Rain', 'Mild', 'High', 'Weak'],
        ['Rain', 'Cool', 'Normal', 'Weak'],
        ['Rain', 'Cool', 'Normal', 'Strong'],
        ['Overcast', 'Cool', 'Normal', 'Strong'],
        ['Sunny', 'Mild', 'High', 'Weak'],
        ['Sunny', 'Cool', 'Normal', 'Weak'],
        ['Rain', 'Mild', 'Normal', 'Weak'],
        ['Sunny', 'Mild', 'Normal', 'Strong'],
        ['Overcast', 'Mild', 'High', 'Strong'],
        ['Overcast', 'Hot', 'Normal', 'Weak'],
        ['Rain', 'Mild', 'High', 'Strong']
    ]

    y = [
        'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
        'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'
    ]

    attributes = list(range(len(X[0])))

    tree = id3(X, y, attributes)

    print("Decision Tree:")
    print(tree)

    test_sample = ['Sunny', 'Cool', 'High', 'Strong']
    print("\nPrediction for", test_sample, ":", predict(tree, test_sample))
