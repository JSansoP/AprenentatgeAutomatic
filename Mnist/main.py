# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import numpy as np


def main():
    X, y = load_digits(n_class=10, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = StandardScaler()
    X_train_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    svm = SVC(kernel="linear", C=1000)
    svm.fit(X_train_transformed, y_train)

    predictions = svm.predict(X_test_transformed)

    correct = np.count_nonzero(predictions == y_test)

    print("Accuracy: ", (correct/len(predictions))*100, "%")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
