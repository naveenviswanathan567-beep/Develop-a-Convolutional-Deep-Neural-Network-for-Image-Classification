# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The objective of this project is to develop a Convolutional Neural Network (CNN) model for image classification. The model classifies fashion images into different categories automatically. It learns important features from input images and predicts the correct class. The performance of the model is evaluated using accuracy, confusion matrix, and classification report.

The dataset used is the Fashion MNIST dataset provided by Zalando Research. It contains 70,000 grayscale images of size 28×28 pixels. The dataset has 60,000 training images and 10,000 testing images. It consists of 10 clothing categories such as T-shirt, Dress, Coat, Shoe, and Bag. The dataset is used to train and test the CNN model for image classification.

## Neural Network Model
<img width="1145" height="802" alt="image" src="https://github.com/user-attachments/assets/9c420395-ecea-40b0-ba58-6c34b29c142d" />


## DESIGN STEPS
### STEP 1: 
Import the required libraries and configure the execution device (CPU/GPU) for building the CNN model.

### STEP 2: 

Load the Fashion MNIST dataset, apply image preprocessing, and create training and testing data loaders.

### STEP 3: 

Design and initialize the Convolutional Neural Network (CNN) architecture with convolution, pooling, and fully connected layers.

### STEP 4: 

Define the loss function and optimizer, then train the CNN model using the training dataset.

### STEP 5: 

Evaluate the trained model using the testing dataset and calculate accuracy, confusion matrix, and classification report.

### STEP 6: 
Test the model with a new image and display the actual and predicted class labels.


### PROGRAM

### DEVELOPED BY :
```
NAME : NAVEEN V
REG NO : 212225240098 
```

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

    
root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)


print("Training Image Shape:", train_dataset[0][0].shape)
print("Training Samples:", len(train_dataset))

print("Testing Image Shape:", test_dataset[0][0].shape)
print("Testing Samples:", len(test_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
class CNNClassifier(nn.Module):
        x = self.pool(torch.relu(self.conv2(x)))

        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x


model = CNNClassifier()

model = model.to(device)

print("\nModel Summary")

summary(model, (1,28,28))


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


def train_model(model, train_loader, epochs=3):

    print("\nName:", NAME)
    print("Register Number:", REGISTER_NO)

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


        print(
            f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}"
        )
train_model(model, train_loader)

    predictions = []
    actuals = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            predictions.extend(predicted.cpu().numpy())

            actuals.extend(labels.cpu().numpy())


    accuracy = correct / total


    print("\nTest Accuracy:", accuracy)


    cm = confusion_matrix(actuals, predictions)


    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=test_dataset.classes,
        yticklabels=test_dataset.classes
    )


    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.show()


    print("\nClassification Report:\n")

    print(
        classification_report(
            actuals,
            predictions,
            target_names=test_dataset.classes
        )
    )
    test_model(model, test_loader)


def predict_image(model, index, dataset):

    print("\nName:", NAME)
    print("Register Number:", REGISTER_NO)


    model.eval()

    image, label = dataset[index]

    image_input = image.unsqueeze(0).to(device)


    with torch.no_grad():

        output = model(image_input)

        _, prediction = torch.max(output,1)


    classes = dataset.classes


    plt.figure(figsize=(4,4))

    plt.imshow(image.squeeze(), cmap="gray")

    plt.title(
        f"Actual: {classes[label]}\nPredicted: {classes[prediction.item()]}"
    )

    plt.axis("off")

    plt.show()


    print("Actual:", classes[label])

    print("Predicted:", classes[prediction.item()])


predict_image(
    model,
    index=331,
    dataset=test_dataset
)

```
### OUTPUT

## Training Loss per Epoch

<img width="228" height="63" alt="image" src="https://github.com/user-attachments/assets/5e1f9c40-32b7-46c2-81ca-d8e3f6e16663" />


## Confusion Matrix

<img width="600" height="510" alt="image" src="https://github.com/user-attachments/assets/26be12fc-02c8-4478-b296-cec151c17440" />


## Classification Report

<img width="607" height="337" alt="image" src="https://github.com/user-attachments/assets/3e0dd568-3f5b-4653-abab-7e150c1f6b55" />

### New Sample Data Prediction

<img width="385" height="406" alt="image" src="https://github.com/user-attachments/assets/6c1eaf24-08ee-4481-9256-e93da8a26537" />


## RESULT
The CNN model was successfully trained and tested on the Fashion MNIST dataset, achieving accurate image classification with performance evaluation metrics.

