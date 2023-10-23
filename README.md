# Term Deposit Subscription Prediction from Bank Marketing Data
> This repository contains the Portugese banking institution marketing campaigns for term deposits dataset, which provides information about client interactions, outcomes, and demographic information. These campaigns were executed through direct means, primarily phone calls, with the primary goal of encouraging clients to subscribe to these financial products. It offers insights into various aspects, including client demographics, contact history, and the results of these campaigns.  The objective is to train and evaluate machine learning models for predicting customer behavior, and to develop and apply strategies to increase term deposit subscriptions.

## General Information
- Dataset analysis work completed in Jupyter Notebook. Link here ([term-deposit-subscription-prediction.ipynb](https://github.com/vbathena/term-deposit-subscription-prediction/blob/main/term-deposit-subscription-prediction.ipynb))
- Dataset in *data* folder
- Images in *images* folder

## Summary of Findings
Based on the data provided, the following key findings have been summarized:

1. Dataset Overview:

    - The data was collected between May 2008 and November 2010, and includes over 41,000 customer records from UCI Machine Learning Bank Marketing dataset [data/bank-additional-full.csv](https://github.com/vbathena/term-deposit-subscription-prediction/blob/main/data/bank-additional-full.csv)
    - The dataset contains 41,188 rows and 21 columns.
    - It contains 11 categorical and 10 numerical features.
    - There are 12 duplicate entries that should be addressed.
   
2. Demographic Insights:

    - The customer base is diverse in terms of age, ranging from 17 to 98 years old.
    - The majority of customers are employed, married, and hold a university degree.
    - Most customers do not have a default on their loans or housing.
    - Cellular contact is the most common communication method.
    - The most common month and day of the week for contact are May and Thursday.
    - A significant proportion of customers have not participated in a previous marketing campaign.

3. Categorical Features:

    - Categorical features include 'job,' 'marital,' 'education,' 'default,' 'housing,' 'loan,' 'contact,' 'month,' 'day_of_week,' and 'poutcome.'
    - The majority of customers are married, have a university degree, and do not have loan or housing defaults.
    - Cellular is the most common contact method, and May and Thursday are the most common contact month and day.

4. Numerical Features:

    - Numerical features include 'age,' 'duration,' 'campaign,' 'pdays,' 'previous,' 'emp.var.rate,' 'cons.price.idx,' 'cons.conf.idx,' 'euribor3m,' and 'nr.employed.'
    - The average customer is 38 years old, with an average loan duration of 287 days and participation in 2.76 campaigns.
   
5. Missing Values:

    - Some features have missing or unknown values, which require handling.
    - The dataset contains 'unknown' values, mainly in 'job,' 'education,' 'default,' 'housing,' and 'loan' columns.

6. Unique Values:

    - The "job", "education" and "month" feature contains too many unique values, which may require further analysis or handling.
   
7. Correlation Insights:

    - Strong positive correlations exist between economic indicators (e.g., emp.var.rate, euribor3m) and negative correlations with customer behavior (e.g., pdays and previous).
    - Features like "poutcome," "emp.var.rate," and "cons.conf.idx" are positively correlated with the target variable.
   
8. Data Imbalance:

    - The class distribution is imbalanced, with a significantly larger number of 'no' class examples compared to 'yes' class examples.
    - Addressing class imbalance is crucial as it can lead to biased model performance.

9. Data Cleanup:

    - The data was cleaned by removing any duplicate rows or rows with missing values and marital column 'unknown' values.

10. Business Objective:

    - The business objective is to find out which clients are more likely to buy the bank’s products over the phone. The bank wants to use different methods to predict this based on the clients’ information and previous marketing results. The bank will compare the methods and choose the best one for future marketing. This will help the bank to sell more products and use their resources wisely.   
   
11. Baseline Model:

    - The baseline accuracy for predicting subscription to term deposits is 0.88741, reflecting class imbalance.
    - Feature overlapping indicates that only 53% of the data can be classified accurately, suggesting room for improvement.
    - The baseline model was a logistic regression model with no feature engineering. The baseline accuracy goal set was 77%.
   
12. Simple Model:   

    - Engineering features selection
       > The following features were selected for the model: age, job, marital, education, default, housing, and loan.
       
    - Data Train/Test Split
       > The data was split into a training set (70%) and a test set (30%).     
       
    - Model
       > A logistic regression model was trained on the training set. The model achieved an accuracy of 0% on the test set.

    - Model scoring
        > - The model was evaluated using the following metrics:
            Precision: 0.90
            Recall: 0.88
            F1-Score: 0.83
            Balanced Accuracy: 0.50
       > - Overall, the model performed better than the baseline on the test set. However, the balanced accuracy of the model is only 0.50, which indicates that the model is not very good at predicting both positive and negative cases equally well.
            
    - Next Steps:
       > - We need to explore other classification algorithms such as K Nearest Neighbors, Decision Trees, and Support Vector Machines.
       > - Hyperparameter tuning and feature engineering could potentially improve model performance.
       > - We need to consider addressing class imbalance using techniques like oversampling, undersampling, or using - different evaluation metrics like AUC-ROC.
       > - Evaluate and compare the performance of these alternative models to determine which one performs the best in terms of test accuracy and other relevant metrics.

11. Model Comparisons:

    - Model Results
    
    | Model | Train Time | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | Inference Time |
    |-------|------------|---------------|---------------|-----------|--------|----------|---------|-----|
    | KNN | 0.017567 | 0.137079 | 0.074928 | 0.826598 | 0.876551 | 0.842047 | 0.526587 | 0.167694 |
    | Logistic Regression | 0.009127 | 0.000000 | 0.000000 | 0.900094 | 0.887420 | 0.834487 | 0.500000 | 0.002627 |
    | Decision Tree | 0.044930 | 0.299784 | 0.101585 | 0.821447 | 0.864466 | 0.838365 | 0.531416 | 0.006046 |
    | SVM | 22.341190 | 0.003705 | 0.002882 | 0.862807 | 0.887582 | 0.835196 | 0.501350 | 1.902650 |

    - Models with Default Settings:
        > - The Decision Tree model has the best performance, with the highest train and test accuracy, precision, recall, F1-score, and balanced accuracy.
        > - The Logistic Regression model has the fastest train and inference times, but the lowest performance.
        > - The SVM model has the slowest train time.
        > - The low accuracy in general suggests that more advanced models or additional feature engineering may be required to improve model performance.

    - Next steps:
       > - Perform more feature engineering, such as creating new features that combine existing features or transforming features in different ways.

12. Improving the Model:

    - Data Cleanup and Feature Selection:
        > - Features were selected or dropped based on their importance and multicollinearity. High importance features include 'emp.var.rate,' 'poutcome,' and 'cons.conf.idx,' while low importance features with high VIF were dropped.
        > - The 'duration' variable was dropped due to its unrealistic use in predictive modeling and its high correlation with the target variable.
        > - SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the class distribution, ensuring equal representation of both classes.
        
    - Model Results:
  
        | Model | Train Time | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | Balanced Accuracy | Inference Time |
        |-------|------------|----------------|---------------|-----------|--------|----------|-------------------|----------------|
        | KNN | 0.194865 | 0.993903 | 0.913771 | 0.852975 | 0.852002 | 0.851900 | 0.852002 | 2.075588 |
        | Logistic Regression | 0.203484 | 0.804542 | 0.807799 | 0.732279 | 0.731766 | 0.731617 | 0.731766 | 0.009409 |
        | Decision Tree | 0.138866 | 0.966049 | 0.906018 | 0.833845 | 0.833836 | 0.833835 | 0.833836 | 0.011962 |
        | SVM | 38.575899 | 0.713415 | 0.719193 | 0.664954 | 0.664656 | 0.664504 | 0.664656 | 7.403238 |

        - KNN Model:
            > - KNN had the best performance of the four models in terms of test accuracy and F1-Score, but it had the high inference time.
        
        - Decision Tree Model:
            > - The decision tree model has the second highest train and test accuracy, precision, recall, F1-score, and balanced accuracy.
            > - It has a relatively short training time and inference time compared to the KNN and SVM models.

        - Logistic Regression Model:
            > The logistic regression model has the shortest inference time, but also the lowest test accuracy and F1-score.

        - SVM Model:
            > The SVM model has the longest train time and high inference time.
        
        Overall, The Decision Tree model is a good choice if we prioritize a trade-off between model performance and computational efficiency.

    - Next Steps:
        > - We have to try using other machine learning algorithms, such as Random Forest or XGBoost.
        > - Perform more feature engineering, such as creating new features that combine existing features or transforming features in different ways.
        > - Use a hyperparameter tuning library to find the optimal hyperparameters for the model.
            

## Technologies Used
- Seaborn
- Pandas
- Numpy
- Matplotlib
- Scipy Stats
- Scikit-learn

## Future Work/Recommendations

Future work could focus on expanding the dataset to improve the accuracy of their term deposit subscription prediction:

- Address missing or unknown values, particularly in the "job" and "marital" columns, which are important features. Consider imputation or data removal.
- Resolve overlapping features to enhance model accuracy.
- Revisit feature selection and engineering to improve model performance.
- Consider using alternative evaluation metrics like ROC-AUC, precision-recall curves, or cost-sensitive metrics to account for class imbalance.
- Use cross-validation to ensure robustness of the model's performance.

## Conclusion
Overall, the Decision Tree model has the best performance, with the highest train and test accuracy, precision, recall, F1-score, and balanced accuracy. It also has a relatively short training time and inference time. The Logistic Regression model has the fastest train and inference times, but the lowest performance. The SVM model has the slowest train time.

Based on models comparison, the Decision Tree model is the best choice for predicting whether a client will subscribe to a term deposit based on their demographic information and the results of previous marketing campaigns.


## Contact
Created by [@vbathena](https://www.linkedin.com/in/vijayabhaskarreddybathena/) - feel free to contact me!
