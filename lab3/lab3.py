import numpy as np
import os

np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGE_DIR = "FIXME"

def save_fig(fig_id, tight_layout=True):
    #path = os.path.join(PROJECT_ROOT_DIR, "images", IMAGE_DIR, fig_id + ".png")
    path = "five" + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    

def random_digit(X):
    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary,
            interpolation="nearest")
    plt.axis("off")

    save_fig("some_digit_plot")
    plt.show()

   
def load_and_sort():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
        #sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    #mnist["data"], mnist["target"]
    return mnist


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    X_train = mnist.data.iloc[reorder_train]
    y_train = mnist.target.iloc[reorder_train]
    X_test = mnist.data.iloc[reorder_test + 60000]
    y_test = mnist.target.iloc[reorder_test + 60000]
    return X_train, y_train, X_test, y_test
    


def train_predict(some_digit, X_train, y_train):
    import numpy as np
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

    # Example: Binary number 4 Classifier
    y_train_4 = (y_train == 4)
    y_test_4 = (y_test == 4)

    from sklearn.linear_model import SGDClassifier
    # TODO
    # print prediction result of the given input some_digit
    y_train_target = (y_train == some_digit)
    sgd = SGDClassifier(random_state = 13)
    sgd.fit(X_train, y_train_target)
    return sgd
    
    
def calculate_cross_val_score(sgd, X, y):
    # TODO
    from sklearn.model_selection import cross_val_score
    print('Cross-Validation Scores: ', cross_val_score(sgd, X, y, cv=3, scoring='accuracy'))

def plot_digit(some_digit):
    """
    Plots the supplied digit.
    """
    # reshape as a 28x28 image, each feature being a single pixel's intensity (0-255)
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, cmap='binary')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':

    X_train, y_train, X_test, y_test = sort_by_target(load_and_sort())
    sgd = train_predict(5, X_train, y_train)
    calculate_cross_val_score(sgd, X_train, y_train)