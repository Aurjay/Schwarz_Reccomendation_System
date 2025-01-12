![image](https://github.com/user-attachments/assets/0128a64f-ae44-4c60-8d1a-f0cf4219646c)

# E-Grocery Store Product Recommendation System

This project demonstrates the development of a personalized **Product Recommendation System** for an **E-Grocery Store** based on user purchase history.

## 1. Introduction

This project aims to provide personalized product recommendations based on previous user purchases. The system uses **Collaborative Filtering** and **Neural Network-based methods** to predict the next product a user is likely to buy.

## 2. Dataset Collection

**Source**:  
The dataset used is from Kaggle's Hunter’s E-Grocery-Stores dataset, containing over **2 million purchase records**.  
**Dataset Link**: [Kaggle Dataset](https://www.kaggle.com/dataset)

## 3. Data Analysis

Data analysis can be found under the **Data Analysis File**.

## 4. Algorithm Selection

**Collaborative Filtering**:
- Recommends products based on users' purchase history.

**Neural Network-based Recommendation**:
- Predicts the next product using a neural network trained on user purchase history.

**Final Model**:  
The Neural Network-based approach was selected due to better performance.

## 5. Model Training

To train the model:
1. Install the dependencies listed in `requirements.txt`.
2. Run `train.py` to train the model.

## 6. Model Deployment

### 6.1 Backend (Flask)
- Flask API for processing requests and providing recommendations.

### 6.2 Frontend (ReactJS on Vercel)
- ReactJS frontend to interact with the recommendation system.

### 6.3 Model Hosting (Azure & Google Cloud)
- The image container is hosted on **Microsoft Azure** and **Google Cloud Platform** using Docker for scalability.

## 7. Conclusion

This project implements a personalized product recommendation system using machine learning models, a Flask backend, and a React frontend.
