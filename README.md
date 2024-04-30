# Machine Predictive Maintenance Classification

## Overview

This project aims to perform multiclass classification on a dataset related to machine predictive maintenance. The dataset contains various features related to machines and their operating conditions, along with labels indicating the maintenance category they belong to.

The main goal is to build a predictive model that can accurately classify the maintenance category of machines based on their features. This can help in proactive maintenance planning and reducing downtime by identifying potential issues before they escalate.

## Dataset

The dataset used in this project can be found on Kaggle: [Machine Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

It consists of the following files:
- `predictive_maintenance.csv`: Contains the data with features and corresponding labels.

## Approach

The classification task will involve several steps, including data preprocessing, feature engineering, model selection, training, and evaluation. Here's an outline of the approach:

1. **Data Preprocessing**: Clean the data, handle missing values, and perform any necessary transformations.
2. **Feature Engineering**: Extract relevant features and possibly create new ones to improve model performance.
3. **Model Selection**: Experiment with different classification algorithms such as Random Forest, Support Vector Machines, Gradient Boosting, etc.
4. **Training**: Train the selected models on the training data.
5. **Evaluation**: Evaluate the models using appropriate metrics such as accuracy, precision, recall, and F1-score. Tune hyperparameters as necessary.
6. **Prediction**: Make predictions on the test data using the trained model..

## Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install them using pip:

`pip install pandas numpy scikit-learn matplotlib seaborn`


## Usage

1. Clone this repository:

`git clone https://github.com/Adamixa/Predecitve_maintenance.git`


2. Navigate to the project directory:

`cd machine-predictive-maintenance-classification`


4. Run the Jupyter notebook to execute the classification pipeline.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

[Adam Adeeb Al-Abduljabbar]

## Acknowledgements

- Dataset provided by Shivam Bansal on Kaggle.
- Done for T5 Bootcamp in Tuwaiq Academy 

