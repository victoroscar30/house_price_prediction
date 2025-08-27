# California Housing Prices Prediction 🏠💰
This project is an Exploratory Data Analysis (EDA) and Machine Learning study aimed at predicting the median house prices in California. The work uses a dataset available on Kaggle and follows the methodology of a comprehensive tutorial on data science and machine learning in Python.

## Project Overview
The main goal of this project is to build a predictive model capable of estimating the median value of houses in California based on characteristics like location (latitude and longitude), house age, number of rooms, population, and median income. The workflow includes the following steps:

* **Exploratory Data Analysis (EDA)**: Understand the dataset, check for missing values, and visualize feature distributions using histograms.

* **Data Preprocessing and Cleaning:** Handle null values (by removing them), remove irrelevant columns, and transform categorical variables like ocean_proximity into numerical values using one-hot encoding.

* **Feature Engineering**: Create new features from the existing ones, such as the bedroom_ratio and household_rooms, to improve model performance.

* **Data Splitting**: Separate the data into training and testing sets to fairly evaluate the model's performance.

* **Model Training**: Train a Linear Regression model to establish a performance baseline, and then a more advanced RandomForestRegressor for better accuracy.

* **Hyperparameter Optimization**: Use GridSearchCV to find the best combination of parameters for the RandomForestRegressor model.

## Dataset
The project uses the housing.csv dataset, which contains housing price data for California. The notebook reads this file from a datasets directory.

## Requirements
To run this notebook, you will need the following Python libraries installed. You can install them using `pip`:

```Shell
pip install pandas numpy matplotlib seaborn scikit-learn
```
## How to Run
**1.** Clone this repository to your local machine.

**2.** Download the `housing.csv` file from Kaggle and place it in a folder named datasets in the root of the project.

**3.** Open the `housing-prices-prediction.ipynb` notebook in a Jupyter environment (Jupyter Notebook, JupyterLab, or VS Code).

**4.** Run the cells sequentially to reproduce the analysis and model training.

## References & Credits
This project was inspired by the NeuralNine YouTube tutorial:

Tutorial Video: [House Price Prediction in Python - Full Machine Learning Project](https://www.youtube.com/watch?v=fATVVQfFyU0)

YouTube Channel: [NeuralNine](https://www.youtube.com/@NeuralNine)
