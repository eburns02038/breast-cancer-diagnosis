# Overview
- According to the National Institute of Health, breast cancer is the most commonly diagnosed cancer among women in the US, and each year 30% of all newly diagnosed cancers are breast cancer. 
- Current methods of diagnosing tumors as malignant or benign predominantly use ultrasound or mammogram imaging to examine texture and measurements of other morphological features to analyze tumors, but the proportion of how much each measurement contributes to the diagnosis varies from doctor to doctor. Additionally, visual examination of images has limited objectivity, and this leads to less accurate diagnoses. 
- Using machine learning algorithms to make a model for determining whether tumors are malignant or benign can increase the accuracy of diagnosis, and reduce time and resources spent on less accurate methods

# Data Analyzed
- The dataset I used for this project was 30 measurements taken from 569 breast cancer tumors classified as either malignant or benign. Some of the measurements included in the dataset were area, perimeter, smoothness, texture, symmetry, compactness, and concavity. Measurements were recorded in 3 groups for each measurement: mean measurements, standard error, and worst measurements. 

# Project Goals
Research Questions
- The research questions that I wanted to address with my model were: 
- 1) What measurements are the most important for diagnosing tumors as benign or malignant? 
- 2) What combinations of measurements in what proportions give the most accurate predictions?
- I wanted to address these questions because establishing a set of measurements that can most accurately diagnose tumors as malignant or benign would increase the efficiency with which diagnoses can be made, and also limit the number of misdiagnoses. 

# Dataset Exploration
- The first thing I wanted to look at was the distribution of the target variable to check how balanced it was. I found that the distribution was slightly imbalanced, with ~60% of the data being for benign tumors and ~40% being for malignant tumors. Ideally for a binary model we would want the target to have a 50/50 distribution between the two possible outcomes so the model isn’t biased towards one outcome of the other, but in this instance having a 60/40 distributed target variable is 1) not a very severe imbalance, and also 2) it better reflects the reality of the data the model would encounter in the real world, where the vast majority of tumors will be benign and only about 10% are malignant. Just in case, I did resampling using cross validation for each of the models to limit the amount of bias towards the more predominant outcome. 
- Only a couple of the features had a normal distribution, but when considering the type of model I was likely to build I did not anticipate needing data to be normally distributed, so I continued with the non-transformed data. 
- Something else that I noticed in the distribution for the various features was that many had higher end outliers. To check whether these would cause problems for the model, I looked at scatter plots for each feature split into the positive and negative diagnosis and found that for the majority of the features, the high end outliers correlated with the positive diagnosis for malignant tumors, so these outliers should be fine to leave in because things like higher measure of compactness or larger area are indicators of a malignant tumor. For a couple of features there were one or two points for benign diagnosis that were high outliers, but since the other measurements for those tumors aligned better with the measurements we would expect for benign tumors, I decided to leave them in and just make note of which measurement had these unusual points, so I could address them later if I ended up using that particular feature in the model. 

# Data Preprocessing
- To ensure that my data would be as clean as possible before starting any analysis, I evaluated the outliers, like I mentioned on the previous slide, and examined the variables for potential errors 
- Next, I checked to see if any data was missing or null. There was one feature in the dataset that was entirely nulls, so I removed that. There was no other data missing in the dataset
- Lastly, in preparing the data for analysis, most of the data was already numeric data, but the diagnosis was a string, with M for malignant and B for benign, so I changed that to binary instead with 1 being malignant and 0 being benign

# Correlation Matrix
- For the initial correlation matrix, I used all of the features in the dataset to see which of the categories of measurements (mean measurements, standard error of measurements, or worst measurements) was the most highly correlated with the diagnosis target variable, and to see if any of the features were highly correlated with each other. 
- The resulting correlation matrix was very large, but I did see a good spread of colors, from things that aren’t correlated at all and areas that are highly correlated, which is good because that tells me that there are a good number of variables that would be useful for the model, and also some that I can remove without really affecting the performance of the model.
- To start with I wanted to focus just on how each of the groups of features correlated with the target, so looking just at the row for the correlation of each of the features with the diagnosis variable, I split the features into the 3 groups of measurements and then checked how many of the features were over 70% correlated with the target. The mean measurements had 5 features that were over 70% correlation, the standard error measurements had none, and the worst measurements had 4. Since I wanted to start by using the set of features that had the most number of highly correlated features to the target, this comparison between the groups indicated that the group that was likely to be most useful for building the model was the mean measurements. 
- One other consideration that I took into account before moving forward with the mean measurements group was looking at the remaining variables that were less than 70% correlation with the target. For the mean measurements vs the worst measurements, it does seem like most of the remaining features are more highly correlated than the remaining features of the mean measurements, which made me second guess whether the mean measurements would actually be the better group to use for the model. 
- To further explore this, I compared the correlation matrix for each of the sets of features with themselves, and found that while the additional features for the worst measurements had generally higher correlations than the mean features, the correlation matrix showed (with the higher proportion of lighter colors) that the worst features were also more highly correlated with each other, which would cause complications for the model later, so I decided to continue with the mean measurements for building the model

# Modeling Process
Overview of models used
- Support Vector Classifier
- Random Forest Classifier
- Gradient Boosted Classifier

Data Processing and Feature Engineering
- Once I had chosen a set of features to start with, I looked at the correlations between them and determined which features were highly correlated with each other and out of those which ones to keep for building the model and which to drop (for example: the measurements for radius, perimeter, and area are almost 100% correlated with each other because they are dependent measurements, so we really only need to keep one of them for the model) 
- After determining a “short list” of 6 features for building the model, I built an initial model that used all of them and then I tuned the hyperparameters of the model to determine which combination would work best for this list of features
- After tuning the hyperparameters, I would run the model, measure the importance of each feature to that iteration of the model, and then drop the feature with the lowest importance and run the model again, repeating this process until I saw the model’s predictive power start to decrease. The point where the model performed best indicated which combination of features would give the most accurate predictions for this model type
Training + Evaluation Methods
- For training all of my models I used an 80/20 split, with 80% of the data being used for training and 20% used for testing. 
- I used cross validation for each model, which runs the model multiple times each time using a different sampling for the test data, to ensure that the results I am seeing are not due to chance, confirm consistency of the algorithm being used, and avoid overfitting the model to the training data
- To evaluate the predictive power of the model, I used the measurement of area under the curve (AUC), which is used to assess the accuracy of a model in predicting binary outcomes, with a higher AUC indicating a higher predictive power, and a lower AUC indicating low predictive power closer to random chance.

# Model 1: SVC
Why choose this model type?
- I chose the SVC model to start with because this model type is less influenced by outliers, which I have a few of, and would also be a good starting point to see what the boundaries are between the malignant and benign datasets
Overview of tuning and feature selection for this model
- For this model, I tried 4 versions each with different kernel types, then chose the one with the highest predictive power to move forward with (the linear kernel).
- Interestingly the accuracy of this model did not improve with iterations of removing the least correlated features. The final version of this model was very good at predicting the negative diagnosis, but not as good at predicting a positive diagnosis, and for this kind of data we would want to prioritize correct positive diagnoses, as these would be more critical to diagnose correctly.
Strengths and weaknesses
- While this model had good overall accuracy, this model type works best when there is a clear separation between the classes, and when revisiting the scatter plots for each feature, there is a fair bit of overlap between the benign and malignant data for many of the features, so this model type may not be the best for this data. 

# Model 2: Random Forest
Why choose this model type?
- I chose the random forest classifier because it has a low risk of overfitting and clearly shows which features are important to the prediction. 
Overview of tuning and feature selection for this model
- For this model type, I tried 5 different iterations to tune the hyperparameters and found that the best hyperparameters were repeating for 10 decision trees, with each tree having a maximum depth of 4
- Next, I calculated the importance of each feature to the model and removed the lowest importance feature until the model didn’t perform as well. Through this method I found that the best combination of features to be included in this model were the mean measurements for texture, compactness, area, and concave points. 
Strengths and weaknesses
- This model had very good scores for the average accuracy and AUC, and had much improved scores for the percentage of correct positive diagnoses, and having a short list of features means the model is less complicated than one that has many features.
- A weakness of this model is that this model type is more computationally expensive than a simpler model type

# Model 3: Gradient Boosted Classifier: 
- Why choose this model type?
- I chose the gradient boosted model because like the random forest it has a high predictive power and shows which features are important to the prediction, but it also learns from each decision tree it builds and tries to build a better decision tree each time, whereas in Random Forest classifiers each decision tree is independent. It would be interesting to see how this affects the accuracy of the model. 
Overview of tuning and feature selection for this model
- For this model, I found the best hyperparameters to be 300 iterations of decision trees with a max depth of 3 per tree. Next, I calculated the importance of each feature to the model and removed the lowest importance feature as I did for the previous model. Through this method I found that the best combination of features to be included in this model were the mean measurements for texture, area, and concave points. 
Strengths and weaknesses
- This model had very good positive and negative predictions, like the random forest classifier, and also had high average accuracy and predictive power. 
- A weaknesses for this model is that, like the random forest model, it is computationally expensive.

# Model Comparison
The initial support vector classifier model did not perform as well as the random forest or gradient boosted models, and the model type did not seem as well suited to the data. The random forest and gradient boosted model types were better suited for the data and were more predictively powerful model types.
Comparing the various evaluation metrics that we looked at, both the Gradient Boosted and Random Forest models performed very well, but the Random Forest model has higher scores for everything but True Negatives. Since we want to prioritize true positives as mentioned before, having a slightly lower score for the true negatives is acceptable. Overall the Random Forest appears to be the better model for this data.

# Conclusions
- Returning to the research questions that we wanted to address at the beginning: “What measurements are the most important for diagnosing tumors as benign or malignant?” and “What combinations of measurements give the most accurate predictions?”
- From the process of building and evaluating all of the models, I determined that the most important measurement for diagnosing tumors is the concave points (or the number of indentations found on the border of the mass). Other measurements of importance were the area, compactness, and texture, and it is the combination of these four measurements that give the most accurate predictions for diagnosis. 
- Benefits to using this machine learning model for breast cancer diagnostics include needing fewer measurements to make an accurate diagnosis (so instead of all 10 of the original measurements, you would only need 4). Machine learning models also increase objectivity of diagnosis and reduces time spent deliberating diagnosis that may not be immediately obvious. Lastly, with a more objective system, using a machine learning model would reduce the percentage of misdiagnoses, which is especially critical for misdiagnosed malignant tumors. 

# Future Goals
- Make versions of the model that use the worst measurements instead of the mean measurements, and see how the importance of the features varies in those models compared to the best model using the mean measurements. 
- Potentially examine model performance on different kinds of cancers, depending on morphological similarities or differences between the cancers and resulting tumors, and determine which other kinds of cancers this model can be applicable to
- Expand on current model to see if I can improve scores:;
- - XG Boost compared to gradient boosted classifier
  - Light GBM model type
  - analyze exactly how much each feature contributes to the final model (how much is each measurement driving predictive power? Try Shapley Analysis)


https://www.cancer.org/content/dam/cancer-org/research/cancer-facts-and-statistics/breast-cancer-facts-and-figures/breast-cancer-facts-and-figures-2019-2020.pdf

https://seer.cancer.gov/statfacts/html/breast.html

https://www.cancer.org/research/cancer-facts-statistics/breast-cancer-facts-figures.html 


