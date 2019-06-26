import zipfile


# zip file inserted into Python

with zipfile.ZipFile("FILE_name.zip", 'r') as zip_ref
    zip_ref.extractall(". ")


# load names (reviews) files and labels files inside the ZIP file into the memory

with open("FILE_NAMES.txt") as f:
    FILE_NAMES = f.read().split("\n")

with open("FILE_LABELS.txt") as f:
    FILE_LABELS = f.read().split("\n")

FILE_NAMES_tokens = [FILE_NAME.split() for FILE_NAMES in FILE_NAMES]

# load the module to transform names(reviews) inputs into binary vectors
# What does this is MultiLabelBinarizer
# Get one-hot encoding of FILE NAME tokens
from sklearn.preprocessing import MultiLabelBinarizer

one_hot_encoding = MultiLabelBinarizer()
one_hot_encoding.fit(FILE_NAMES_tokens)

# need to divide the data into training and test

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(FILE_NAMES_tokens, FILE_LABELS, test_size = 0.2, random_state = None)

# Create SVM classfier
# What does this is LinearSVC

# and then Train it

from sklearn.svm import LinearSVC

_svm = LinearSVC()
_svm.fit(one_hot_encoding.transform(X_train), Y_train)




