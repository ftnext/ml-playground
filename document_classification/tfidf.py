import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class DocumentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


categories = [
    "alt.atheism",
    "soc.religion.christian",
    "comp.graphics",
    "sci.med",
]
newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)
newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)
print("#train:", len(newsgroups_train.data))
print("#test:", len(newsgroups_test.data))

vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)
print("shape train:", vectors_train.shape)
print("shape test:", vectors_test.shape)

X_train = torch.from_numpy(vectors_train.toarray()).float()
Y_train = torch.tensor(newsgroups_train.target).long()
X_test = torch.from_numpy(vectors_test.toarray()).float()
Y_test = torch.tensor(newsgroups_test.target).long()

model = DocumentClassifier(len(vectorizer.vocabulary_), 128, len(categories))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

epochs = 15
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    Y_pred = predicted.numpy()
    print(confusion_matrix(Y_test, Y_pred))
    print(precision_recall_fscore_support(Y_test, Y_pred))
