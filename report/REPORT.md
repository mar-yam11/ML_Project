# Real-Time Material Classification Using SVM and k-NN

## 1. Introduction

This project is a simple machine learning app for material classification based on feature extraction techniques and classifiers.

## 2. Dataset and Augmentation

The dataset contains six material classes. Data augmentation techniques such as rotation, flipping, and brightness adjustment were applied to increase the dataset size.

## 3. Feature Extraction

Images were converted into fixed-length feature vectors using Histogram of Oriented Gradients (HOG) combined with color histograms.

## 4. Classification Models

Two classifiers were implemented: Support Vector Machine (SVM) and k-Nearest Neighbors (k-NN). SVM achieved better performance.

## 5. Unknown Class Handling

A simple threshold-based rejection mechanism was used to classify uncertain inputs as an Unknown class.

## 6. Results

The SVM classifier achieved approximately 78% validation accuracy, while the k-NN classifier achieved around 77%.

## 7. Real-Time System

The trained SVM model was integrated into a real-time application using OpenCV to classify live camera frames.
