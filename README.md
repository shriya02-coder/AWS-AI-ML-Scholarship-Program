# Project: Use-a-pre-trained-classifier-to-identify-dog-breeds

## Project Goal
The goal of this project is to enhance your Python programming skills by using a pre-developed image classifier to identify dog breeds. The focus will be on applying Python tools to utilize the classifier effectively.

## Project Description
Your city is hosting a citywide dog show, and you have volunteered to assist the organizing committee with contestant registration. Each participant must submit an image of their dog along with biographical information. The registration system tags these images based on the provided biographical information.

However, some participants may register pets that are not dogs. Your task is to use the provided Python classifier to ensure that only dogs are registered for the show.

### Key Objectives:
1. **Determine the Best Classification Algorithm**: Use your Python skills to evaluate which image classification algorithm (AlexNet, VGG, or ResNet) performs best in classifying images as "dogs" or "not dogs."
2. **Evaluate Breed Identification**: Assess how well the best classification algorithm identifies specific dog breeds.
3. **Measure Performance**: Time how long each algorithm takes to classify images and consider the trade-off between accuracy and runtime.

## Classifier Function
You will be using a convolutional neural network (CNN) for image classification. CNNs are well-suited for detecting features in images, such as colors, textures, and edges, and then using these features to identify objects within the images. The CNNs you will use have already been trained on a large dataset (ImageNet) containing 1.2 million images.

The classifier function is provided in the `classifier.py` file. The `test_classifier.py` file demonstrates how to use the classifier function.

### CNN Architectures:
- **AlexNet**
- **VGG**
- **ResNet**

## Similar-Looking Breeds
Some dog breeds look very similar, and distinguishing between them can be challenging for image classifiers. The breeds that are often confused include:
- Great Pyrenees and Kuvasz
- German Shepherd and Malinois
- Beagle and Walker Hound

## Tasks
1. **Load and Preprocess Images**: Use Python to load and preprocess the images for classification.
2. **Classify Images**: Apply the classifier function to determine if the images are of dogs or not.
3. **Evaluate Accuracy**: Assess the accuracy of each CNN architecture in classifying the images.
4. **Measure Runtime**: Time how long each CNN architecture takes to classify the images.
5. **Compare Results**: Analyze and compare the accuracy and runtime of the different architectures to determine the best one for this application.
