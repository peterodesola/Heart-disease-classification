# Heart-disease-classification using Machine Learning (Ensemble method)
## Table of Contents
- [INTRODUCTION](#INTRODUCTION)
- [BACKGROUND](#BACKGROUND)
- [RELATED WORKS](#RELATED-WORKS)
- [JUSTIFICATION](#JUSTIFICATION)
- [EXTREME GRADIENT BOOST](#EXTREME-GRADIENT-BOOST)
- [PROBLEM DEFINITION](#PROBLEM-DEFINITION)
- [PURPOSE AND RESEARCH QUESTIONS](#PURPOSE-AND-RESEARCH-QUESTIONS)
- [AIM AND OBJECTIVES](#AIM-AND-OBJECTIVES)
- [PROPOSED ARTEFACT AND SOCIETAL IMPACT](#PROPOSED-ARTEFACT-AND-SOCIETAL-IMPACT)
- [RESOURCES AND PROJECT IMPLEMENTATION](#RESOURCES-AND-PROJECT-IMPLEMENTATION)
- [METHODOLOGY](#METHODOLOGY)
   - [Data Collection](#Data-Collection)
   - [Exploratory data analysis](#Exploratory-data-analysis)
   - [Data pre-processing](#Data-pre-processing)
- [MODEL DEVELOPMENT](#MODEL-DEVELOPMENT)
- [ALGORITHM SELECTION](#ALGORITHM-SELECTION)
- [MODEL FITTING](#MODEL-FITTING)
- [MODEL COMPLEXITY](#MODEL-COMPLEXITY)
- [HYPER-PARAMETER OPTIMISATION](#HYPER-PARAMETER-OPTIMISATION)
- [CROSS VALIDATION](#CROSS-VALIDATION)
- [ENSEMBLE CLASSIFICATION APPROACH](#ENSEMBLE-CLASSIFICATION-APPROACH)
- [MODEL EVALUATION](#MODEL-EVALUATION)
- [CONFUSION MATRIX](#CONFUSION-MATRIX)
- [PRECISION AND RECALL](#PRECISION-AND-RECALL)
- [F1 SCORE](#F1-SCORE)
- [ACCURACY](#ACCURACY)
- [ROC AUC](#ROC-AUC)
- [SOFTWARE DEVELOPMENT](#SOFTWARE-DEVELOPMENT)
- [DISCUSSION](#DISCUSSION)
- [MODEL TUNING](#MODEL-TUNING)
- [Hyperparameters](#Hyperparameters)
- [ROC](#ROC)
- [PERFORMANCE COMPARISON AGAINST BENCHMARK STUDIES](#PERFORMANCE-COMPARISON-AGAINST-BENCHMARK-STUDIES)
- [LIMITATIONS](#LIMITATIONS)
- [CONCLUSION AND FUTURE WORK](#CONCLUSION-AND-FUTURE-WORK)
  


## INTRODUCTION
The heart is known to be a muscular organ that supplies blood through the blood vessels of the circulatory system. Heart disease are range of diseases that affect the functions of heart, and such diseases could be coronary heart disease, heart disease, heart attack, stroke, heart failure, heart arrhythmia, heart valve complications, rheumatic heart disease and other heart conditions. World health organisation (WHO) study reports cardiovascular diseases as the world leading death rate cause with 17.9 million, 39 percent of global deaths annually (WHO 2021.). Cardiovascular disease requires immediate and special attention as it is one of the world most deadly disease associated with the largest and most relevant organs in the human body. Most of the medical illness are associated with the heart and for this reason, it is of great necessity to predict cardiovascular disease and making a comparative research topic a priority. Many factors are responsible for the cause of heart disease and has made medical diagnosing difficult (Alqudah 2017, 215-222). Due to inability of the patient to be diagnose because of lack of instruments accuracy, 20-40% patients had heart disease (McClellan 2019.). As a result of this, there is need to have further study on efficient algorithm to be used for cardiovascular prediction. It is very critical to develop an effective and ideal model for heart disease prediction to aid diagnosis and medical significant which various study have been done in the past (Pouriyeh et al 2017.)
## BACKGROUND
Cardiovascular disease has been the main cause of highest death rate globally (WHO 2021), and early detection of these diseases will save lives. Prediction of cardiovascular disease is regarded as one of the most important subjects in clinical data analysis. The amount of data in the healthcare industry is huge. Data mining turns the enormous collection of raw healthcare data into information that can help to make informed decisions and predictions (Shalev 2020). Cardiovascular prediction can be of good advantage for medical personnel for fast diagnosing and more precise accurate results by the implementation of computerised classification and cardiovascular disease prediction. Conventional method proves to have deficient performance when compared to the recent fastidious evolving artificial intelligence techniques of approach. For this proposed study, dataset would be collected from Kaggle with 1025 observations and 14 relevant features. Proper data pre-processing would be conducted to avoid skewness or data imbalance for effective performance of the model implementation, likewise, feature selection shall be considered to see how relevant the features are in the prediction. 
Data mining have been successfully applied in many fields such as manufacturing engineering, web mining, mobile computing, marketing, customer relationship management, fraud detection, intrusion detection, lie detection and bio informatics. Data mining has been successfully applied in healthcare fraud and detection of abused cases. Analysis of data has proven to be significant in health sector. 
This study proposes the implementation of improved, accurate and efficient classification of heart disease predictive system using a data mining knowledge. Health experts can use the implemented knowledge to deliver quality services to the service users.
## RELATED WORKS
Research studies have been conducted on cardiovascular disease prediction based on risk assessment and classification combining the application of both machine and deep learning, and this study keened on how to make improvement in the accuracy on previous studies done which has been a motivation for improvement. 

Suri et al 2022, stated in their recent research study on powerful paradigm for heart risk stratification with the application of multilabel, multiclass, and ensembled-based machine learning paradigms that conventional method of approach to cardiovascular risk stratification showed inferior result regarding performance compared to the newly and fast-moving artificial intelligence approach. Implementation of most recent paradigms cardiovascular risk assessment was proposed, which were multiclass, multi-labelled and ensembled methods in office based and stress test laboratories. A total sum of 265 cardiovascular based studies were used by implementation of preferred reporting items to be reviewed systematically and meta-analysis (PRISMA) model application. The three methods were reviewed comprehensively based on some attributes such as architecture, pro-and-cons, application, scientific validation, clinical evaluation, and bias risk. The multiclass and ensemble performance metrics used were accuracy, sensitivity, specificity, AUC, F1-score. F1-score was solved using true positive, false positive, true negative and false negative while label-based technique performance evaluation parameters were checked on each label based on true positive, false positive, true negative and false negative. Artificial intelligence based for cardiovascular based risk assessment were successful and promising when using the three cardiovascular disease paradigms in a non-cloud and cloud-based frameworks.

Uddin et al, 2021 conducted study on ensemble method based multilayer dynamic system to predict cardiovascular disease using machine learning. The three classifier models combined to form the ensemble method for classification are random forest, Naïve Bayes and the K Nearest Neighbor (KNN). Also, ensembled method-based multilayer dynamic system (MLDS) which have the purpose of increasing knowledge in every layers. The model that was proposed applied Correlation attribute evaluator (CAE), Gain Ratio Attribute Evaluator (GRAE), Information Gain Attribute Evaluator (IGAE), Lasso and Extra Trees Classifier (ETC) for feature selection. Dataset was collected from Kaggle with 70,000 instances, the models achieved accuracy of 88.84%, 89.44%, 91.56%, 92.72% and 94.16% respectively in respect to the trained and test data splitting ratios of 50:50, 60:40, 70:30, 80:20 and 87.5:12.5. The model also had 0.94 AUC which is 94% probability of classifying correctly positive and negative classes. The outcome of the model had satisfactory performance with high accuracy, but a high recommendation was made for future study considering the use of neural network model.
## JUSTIFICATION
In hospitals, most decisions made are based on medical practitioners’ insight and experience when there is knowledge rich information hidden in the datasets or hospital records. Such practice eventually leads to error, unwanted biases, reduction in quality service provision and excessive cost for the patients. It has been proposed to implement a heart disease predictive system using artificial intelligence (AI) which will integrate clinical decision support based on patients’ medical records (datasets). The implementation of this system will reduce medical errors, increase patients’ safety, decrease bias and improve patient’s outcome. As it is to be noted that right heart disease prediction saves life, while wrong prediction takes lives.  Therefore, implementation of heart disease predicting system with well improved accuracy can bring solution to early death and reduction to statistical death rate annually stated by world health organisation. Data mining have a great ability and capability to extract a knowledge rich record that can be used to significantly improve the quality of clinical decisions.
Several studies had been conducted on implementation of a predictive system with accuracy and a lots of machine algorithms had been used by which they all had a good accuracy, while recommendation on accuracy improvement were emphasized to enhance performance. 
Many algorithms had been used in the previous studies while improvement of accuracy was still emphasized for a good predictive system to be implemented. This study will make use of Random Forest (weak classifier) and Extreme gradient boosting classifier (strong classifier) to improve the accuracy of random forest classifier because of its ability to support other types of tree ensemble algorithms like Random Forest. Extreme gradient boost classifier would be used as predictive model because of its ability. 
### EXTREME GRADIENT BOOST
Extreme gradient boost is a supervised classifier that implement gradient boosted decision trees that is designed for speed and better performance while Random Forest is a machine learning algorithm used for both classification and regression and function by collection of trees and each ensembled trees comprise of data sample extracted through training set with replacement called bootstrap sample. XGBoost had been used in so many supervised classification problems in which it came out with an impressive performance.
#### FEATURES OF XGBOOST
-	It can combine several optimisation techniques to achieve perfect result in a short span of time. 
-	It can avoid overfitting with a well-managed regularisation ability. 
-	XGBoost build a tree at a time so each data relating to the decision tree is taken to account data filling takes place in case of missing data, which helps in the implementation of gradient algorithms in combination with decision trees algorithm for better accuracy.
-	It has trees learning algorithm and at the same time linear model solver, and it gives substantial number of hyperparameters, which many of them requires tuning for imbalance classification to get the best out of algorithm on the datasets. 
-	XGBoost has been trained to reduce loss function while the “gradient” boosting has been defined to be the steepness of the loss function such as the number of errors. Designed to correct errors made by the existing state of the ensemble of decision trees.
Reviewing XGBoost performances in relation to study conducted on binary classifications, XGBoost has been chosen for the purpose of this study to help improve accuracy and performance of the predictive system as related works proved it is a perfect algorithm for binary classification improvement.

S. Pawar et al in 2021, conducted a study on breast cancer prediction with machine learning using the following algorithm (XGBoost classifier, Decision Tree, K-Nearest Neighbor, Ada-Boost clasaifier, support vector machine and Random Forest classifier) on Wisconsin Breast Cancer Diagnosis dataset that was collected from UCL machine learning repository. XGBoost had the highest accuracy with 98.24%, while other had good accuracy above 90%.

In 2018, B.E.K. Guzel compared the performance of boosting classifiers on breast thermography images in which breast tissue images were collect with the use of segmentation of breast thermography images. Adaboost, Gradient Tree Boosting and XGBoost classifiers were adopted in performance check, XGboost had the best performance based on feature importance metrics.
Lastly, C.D. Anisha and Arulanand in 2020 had a study on early detection of Parkinson’s disease using ensemble method by which the prediction was based on bagging classifier, Adaptive Boosting classifier, Gradient Boosting Machine classifier and Extreme Gradient Boosting classifier which were implemented by optimal parameters acquired using hyper parameters tuning process. XGBoost came out with outstanding performance base on the following evaluation metric precision and F-1 score.
So many studies mostly for supervised binary classifications proved XGBoost classifier has high classification predictive efficiency because of it features previously stated. It has been successfully used mostly in health sectors; therefore, this study (heart disease prediction) proposed the use of Random Forest as a weak classifier and XGBoost as a supporting algorithm to improve the accuracy and performance of heart disease predictive system. 
## PROBLEM DEFINITION
Cardiovascular disease has been the major cause of life threats globally (WHO, 2021), many lives has been lost to heart diseases due to lack of early diagnosis. The major challenge faced by the medical health sector is inability to detect early stages of problems related to the heart. Detection of cardiovascular disease at early stage will reduce the rate of death statistics globally. For early cardiovascular disease detection, and death rate reduction caused by the heart problem, this study proposed implementation of improved, and accurate heart disease predicting system using machine learning knowledge.
## PURPOSE AND RESEARCH QUESTIONS
In the proposed study, machine learning method would be implemented for heart disease prediction through the extracted medical knowledge history (dataset), and many machine algorithms will be developed to see algorithms with the best performance for the implementation of accurate predictive system. Many machines have been created for cardiovascular disease predictions, but recommendation for more research studies for improved and reliable model accuracy are being emphasized. Further, an approach on how to improve the performance of algorithms base on accuracy for implementation of a heart disease predictive system have been proposed (Mohan, 2021.), and base on this we address the following research questions:
-	How reliable and accurate is a heart disease predictive system using the patient’s risk factors or medical records for prediction?
-	How can the accuracy of heart disease predicting system be improved with machine learning algorithms using patient’s risk factor/historical records?
-	Does feature reduction/ feature selection have any significant changes in the model performance?
## AIM AND OBJECTIVES
## AIM
This study is aimed at building an improved, accurate heart disease predictive system by using the hidden knowledge associated with various patient’s historical records (dataset) collected.
## OBJECTIVES
The objectives are:
-	To critically review previous works done on related topic.
-	To carry out the exploratory data analysis of the heart disease dataset.
-	To implement an algorithm that will accurately predict heart disease at early stage.
-	To implement a friendly graphical user interface (GUI) for easy access to the users for diagnosis.
## PROPOSED ARTEFACT AND SOCIETAL IMPACT
A friendly graphical user interface (GUI) is proposed for this study. Stream lit is found in python language, it is a web application made for data science and machine learning with open-source framework. Stream lit can work perfectly with python libraries.
Development of a graphical user interface for cardiovascular disease prediction would grant the people in the society access to always examine themselves against heart disease symptoms. Development of such predicting system with a friendly easy to use web application will cause reduction in death rates, medical check-up cost and stress and will reduce biased results from medical practitioners.
The proposed software (web application) which would be developed using the stream lit graphical user interface library will have the following features:
1. Feature display: This is where individual will enter their details base on the risk factors used in developing the predicting system.
2. Title page: This will display the name or function of the system (heart disease predicting system)
3. Prediction result display: This where the outcome of the heart disease tests will be displayed.
## RESOURCES AND PROJECT IMPLEMENTATION
The resources needed for the development of heart disease predictive system proposed are basically software.
-	Computer system (Windows 11, RAM: 8GB, Processor Inteli3: 2.59HGz, System type: 64bit)
-	IDE: Pycham
-	Programming language: Python 3.8
-	Heart disease dataset: The dataset was collected through secondary source, it is a combination of five databases (Cleveland, Hungary, Switzerland, Stalog and Long Beach V) based on medical history, and it is in a csv format.
-	Major python libraries needed
Matplotlib
Numpy: For mathematical performance based on arrays
Pandas: Use for loading files in various format
Seaborn: Visualisation function
-	Machine learning algorithms for classification
1. Logistic Regression 
2. Naïve Bayes Classifier
3. Random Forest Classifier
3. Extreme Gradient Boost Classifier
4. K Nearest Neighbour Classifier
5. Decision Trees Classifier
6. Stacking classifier
7. Voting Classifier
Journals/Article: IEEE, Google scholar, Research gate, IJERT, Hindawi, National library of medicine and PARC.
## METHODOLOGY
The heart disease prediction study is proposed to be a quantitative research method, with a scientific approach. The dataset to be use are in numbers (numeric) with 14 features. Two approaches would be implemented in the study for model fitting to compare the performances and to see how reliable and accurate the predicting system would be. The first approach would be addressed with all the 14 features in the dataset, while the second approach would be based on feature selection and the feature would be selected based on how important the features are in the dataset before fitting with the eight algorithm classifiers and checking for performances evaluation. The hybrid ensemble method has been proposed for improve performance of the heart disease predicting system.

![Screenshot 2022-08-18 135245](https://github.com/user-attachments/assets/61c009eb-fa8c-4175-90d9-48a2aeacfcd9)

## Data Collection
Data collection was through secondary source because of limited time frame for the execution of the thesis. The dataset was dated from 1988 and was collected from four data base sources which are the Cleveland, Switzerland, Long Beach V and Hungary. The original dataset contains 76 attributes which include the target variable, the 76 attributes was reduced to 14 by categorising the relevant from the irrelevant attributes using feature selection (John et al, 1994.). While target variable indicates the patients with cardiovascular disease and those that has no heart disease.
The features or risk factors are collected from hospital database based on patients’ medical records. The dataset consists of 14 columns (features) and 1025 rows (observation) which are stored in csv format for machine learning. For this study, seven algorithms will be implemented while the best three algorithms with the best performance will be ensembled for development of an improved heart disease predicting system.
## Exploratory data analysis
Exploratory data analysis is a phase in machine learning that helps data analysts or data scientists to carry out investigation on a data, for easy detection of dataset patterns, and to identify anomalies, to spot hypothesis and to observe assumptions through the help of statistical summaries and graphical visualisation. Exploratory data analysis gives better insight of the dataset (Greiff 2000).
The types of datasets were checked to know the data type of the attributes we are dealing with, and to address them according to their datatypes or nature.

![data info](https://github.com/user-attachments/assets/ab2f7d0d-782d-44ad-828b-e57b8c3176ae)

The exploratory data analysis could be:
- Univariate Analysis: In univariate data analysis, only one variable is analysed. This means one feature, or a column is analysed. Univariate analysis can be graphically executed by histograms, pie chart, box plots, etc.
- 
![sex distribution](https://github.com/user-attachments/assets/890cca25-41a9-485a-97b5-24eb06d5ac51)

![heart disease gender distribution](https://github.com/user-attachments/assets/e2663b68-ff12-4558-b69c-852c2021471a)

- Multivariate data analysis: Multivariate data analysis can be the process of analysing the relationship between two or more variables. Multivariate analysis can be displayed using scatterplot, heat map etc. The data analysis will be carried out using both the univariate analysis and multivariate analysis.

![heapmap](https://github.com/user-attachments/assets/2768a2f8-7bfe-44c7-bd64-e9311a1f31c1)

![scatterplot](https://github.com/user-attachments/assets/dce30503-1097-469d-8b91-c34916e13565)


## Data pre-processing
Data pre-processing is an important phase in data mining where statistical analysis is observed because of inconsistency in the real-world data. Data pre-processing phase involves the transformation of raw or unprocessed data into an understandable data format. This is the stage where fixing or removal of incorrect, corrupt, duplicated, empty and incomplete values are carried out for best performance of the models and improved accuracy. Datasets are known to be messy, and they need to be well examined for enhancement of algorithms and outcome. The following steps were taken in the pre-processing phase:
-	Import the libraries: Important libraries were imported for the machine learning study such as pandas, seaborn, numpy, scikit learn, matplotlib etc.

![libaries](https://github.com/user-attachments/assets/892a35f5-c502-4ff9-870b-243854dca440)

-	Import and load dataset: The heart disease csv dataset was imported for reading using appropriate command.
-	
![dataset loading](https://github.com/user-attachments/assets/11ced0ba-fe44-442e-983b-52b9f0398a38)

-	Checking and handling of missing data or values: Dataset must be checked if there are missing data and must be well handle. It is either removed or replace. There was no missing value recorded, which means all values were imputed.

![checking for missing values](https://github.com/user-attachments/assets/09fa6c6b-2f17-4500-85e2-e4099b416a3c)

-	Checking for duplicates: Duplicate is when a particular information or data is being repeated or same data appearing multiple times. Therefore, duplicates must be observed in dataset to avoid observation from same records of a particular record. The duplicates observed in the dataset were removed appropriately.

![checking for duplicate and handling](https://github.com/user-attachments/assets/00560009-a70c-46f9-a760-2ff904093d77)

-	Checking and handling of outliers: Outliers are the values or observations that lies abnormally to other values in distance and could occur because of data input error, sampling problems, and natural variation. Some outliers demand immediate removal while some could be left as they will not affect performance. (Smith and Martinez, 2011) in their research study on classification improvement in increasing accuracy by detecting and removing observations that are misclassified. The outliers in the dataset were appropriately removed using the interquartile range for effective performance of the models.

![checking for outliers](https://github.com/user-attachments/assets/9aecaee2-48ff-4e80-96e2-4393bbf933dc)

![outliers](https://github.com/user-attachments/assets/51cc19a3-fb8e-4869-b4c3-81a2062ec4e4)

![removing outliers](https://github.com/user-attachments/assets/6cd0d9b6-8403-452a-a696-9cbcb39086b9)

-	Check for categorical values in the data (one-hot encoding): This is the phase where all categorical data were converted to numerical values by applying one hot encoding imported from sklearn pre-processing library for the machine to read. By so doing, machine models can now make their decisions on how the labels can be operated.
- Feature scaling of the data: This is the final pre-processing stage. Feature scaling is one of the important methods used in limiting the range of variables, so each variable can be evaluated on common grounds, no bias. It was applied to independent variables. It was used for normalising the data within a certain range because the dataset is not in gaussian distribution (skewed). It also helped to speed up the calculations in the proposed algorithms (Saranya and Manikandan 2013). Both the trained and test sets were scaled using min max scaler imported from scikit learn library. The scaling output was in an array format which was converted to data frame with the use of panda’s data frame method.    
-	Split dataset into train and test set: Splitting of data is very important to avoid biasness when evaluating predictions. The dataset was split into train and test set, where the trained set was 70% and test was 30%. We had the following parameters for the train and test sets.
X train which happened to be the trained part of the matrix of features.
X test which is the test part of matrix features.
y train was the training part of the dependent variable that is associated to X train.
y test was the test part of the dependent variable that was associated to the X test.
-	Feature importance/selection: Feature importance is necessary for identification of how relevant or influential an attribute is in the data. It is used in dropping irrelevant features for dimensionality reduction and to improve the performance of models (Fisher, Rudin and Dominici 2019). The feature importance was checked using extreme gradient boost classifier (XGBoost Classifier), and the feature importance showed that chest pain (CP) has the highest level of importance, then thalassemia, ca, exang and fast blood sugar has the least relevance.

![feature importance](https://github.com/user-attachments/assets/31b589e5-16c1-4a25-baab-e381baa5d6ea)

## MODEL DEVELOPMENT 
## ALGORITHM SELECTION
Model development is the next stage after dataset have been put to good shape or condition for efficient performance of the models. The algorithms were selected based on the problem to solve. The data is a supervised machine learning, and it is based on classification indicating if heart disease is present or not. It is a binary classification problem with dependent (target), and independent variables. The rules of selecting machine learning algorithms for classification were considered  (Quinlan, J. Ross 2014), likewise for the proposed dataset rules have been generated based on different priority  (Quinlan, J. Ross 1995). Based on algorithm selection rules (Quinlan 2014), K- Nearest Neighbour, Naïve Bayes, Logistic Regression, Decision Trees, Random Forest, Multilayer Perceptron and Extreme Gradient Boosting classifiers have been selected to test their performance by comparing their accuracies, precisions, recalls and F-1 scores to choose the three best performing algorithms for ensemble technique for the final modelling. 
## MODEL FITTING
Model fitting is the phase in which machine learning models are measures on how models generalise to similar data on which they were trained. A well fitted model performs better and generates accurate outcome.
Appropriate data pre-processing (missing values checked, outliers checked and handled, duplicate checked and removed, data skewness considered) was carried out and dataset was put into good condition for better performance, also machine learning rules were considered to get the best of our model performance (Quinlan 2014) All the steps for data pre-processing and data splitting were applied to all the seven proposed algorithms, and dataset was splits into training and test sets (X train, X test, y train, y test respectively) at 70% to 30% ratio, where 70% train and 30% test. Models were all fitted based on individual standard fitting codes.
In the model fitting phase, two approaches were considered for model fitting; the first approach, models were fitted with all the 14 features while in the second approach we selected 8 features to be fitted with the proposed algorithms based on feature importance. For model fitting, seven algorithm classifiers (K Nearest Neighbour, Logistic Regression, Naïve Baye, Multilayer Perceptron, Decision Trees, Random Forest, and Extreme Gradient Boosting) were implemented. 
## MODEL COMPLEXITY
Model complexity can be defined as the function of complexity that is to be learned, which could be like a polynomial degree. Model complexity level can be determined in respect to the nature of the training data. Failure in spreading the amount of data or the entire data uniformly throughout different possible scenarios, then there should be consideration in the model complexity reduction because when there is high model complexity there would be overfitting on a small number of data points. Overfitting is then process of training models that suitably fit the trained data but fails to generalise to other data (Schneider, Xhafa and Linke 2002). 
(M. Pidd 1996) in his study on “The five principles of simulating models”, stated a simple model is preferable to complex ones, simplicity is the essence of simulation. (Ward 1989) stated model simplicity relating it to transparency (relating it to understanding) and constructive simplicity (relating to the model itself) and reinforced the idea of model simplicity in his study.
Model complexity was seriously put into consideration to avoid poor performance of proposed models. Models were tuned to correct errors that might affect performance due to overfitting.
(Bergstra and Klop 1982) stated that the use of column subsampling is much more effective than conventional row subsampling when preventing overfitting. The subsampling of rows was done using hyper-parameter subsample, colsample_bytree while the tree structure was established calculating the leaf scores, regularisation, and the objective functions at each level. The tree structure was used over again in subsequent iterations which will eventually reduce model complexity. Extreme gradient boosting hyper-parameters are many and the hyperparameters can be implemented in carrying out some tasks as the model desires. They have their default hyperparameter for optimisation if not set at the initial, but the parameters can be inputted as desired by chosen model.
## HYPER-PARAMETER OPTIMISATION
Hyperparameter optimisation or tuning in machine learning is the process by which a set of optimal hyperparameters are chosen for a learning algorithm, the values are basically used to control the learning process of algorithms to find the best performing algorithm when evaluating based on validation set (Feurer and Hutter 2018). 
There four common hyperparameter optimisation.
•	Manual search
•	Grid search
•	Random search
•	Bayesian optimisation
Random search is the process by which random combinations of various hyperparameters are utilised in getting the solution for the model built, while the manual hyperparameters are set manually and tuned for model optimisation. The random search optimisation was adopted for this study because of its ability to run the train-predict-evaluate cycle that can be done automatically in a loop of hyperparameter on a pre-decided grid, also the manual search was used for optimisation of some models. Generalisation performance was estimated using cross validation (Bergstra and Klop 1982).
## CROSS VALIDATION
Cross validation is also known to be rotation estimation is a model validation technique that is used in evaluating how statistical analysis result will generalise to data that are independent. Cross validation is known as a resampling technique which utilises different portions of the data that uses different iterations to test and train a model (Kohavi and John 1995). The main goal of cross validation is to test the ability of the model to predict new data that has not been used for estimation, so that overfitting or bias selection can be flagged (Cawley 2010). Cross validation was implemented on the models to improve and evaluate the performance of the model, if over-fitted, underfitted or accurate.
## ENSEMBLE CLASSIFICATION APPROACH 
The ensemble classification technique is the combination of several models to improve the result of a machine learning, this technique influences the production of better predictive performance when compared to single model (Ho, Hull and Srihari 1994). (Dietterich 2002) in his study on sequential data in machine learning proved that ensembles technique can overcome three problems, which are:
1.	Statistical problem: Statistical problem happens when the amount of available data has large hypothesis spaces.
2.	Computational problem: this occurs because of inability of a learning algorithm to guarantee finding the best hypothesis.
3.	Representational problem: The representational problem can be caused when hypothesis space fail to have any good approximation of classes that are target.
For the proposed of this study, two ensembled classifiers were implemented to evaluate their accuracies and individual ensemble technique performance. The two adopted ensemble techniques are:
1.	The stacking classifier approach
2.	The voting classifier 
The two ensemble classifier techniques were used for comparison of results, to evaluate the performance of the two techniques. The ensemble technique was adopted to improve the accuracy and performance of the models. Finally for the purpose of this study three best performing models from the proposed algorithms were ensembled using the stacking classifier and the voting classifier. The three models ensembled were the Decision Trees Classifier, the Random Forest Classifier, and Extreme Gradient Boosting Classifier.
## MODEL EVALUATION
Model evaluation is the process by which machine learning model performance, strength and weaknesses are being checked using different evaluation metrics (Hossin and Sulaiman 2015). Confusion matrix, accuracy score, precision, recall, AUC and F-1 score were used for the model evaluation.
## CONFUSION MATRIX
Confusion matrix is a structured table that contains the true values and predicted values known as true positive and true negative, it can be use in the evaluation of performance of classification models (Vujovic 2021). The target variable comprises of two values, which are positive and negative. 
The columns are the actual values of target variables while the rows are the predicted values of our target variables. The confusion matrix table is divided into four parts, two columns and two rows respectively. 
-	True positive (TP): Here the values are identified as true because they are indeed true.
-	False positive (FP):  The values are predicted as true/ positive in the division, but they are false/ negative.
-	False negative (FN): The values were predicted as false/ negative, but they are true/ positive.
-	True negative (TN): The predicted value is negative while the actual value is negative.
## PRECISION AND RECALL
- Precision explains the number of correctly predicted cases, which eventually turned out to be positive.
Precision mathematical expression:
Precision =      TP/ (TP + FP) ------------------------(3)
This is designed to know if our model is reliable or not.
- Recall is based on the exact number of the actual positive cases that was able to be predicted correctly with our models.
Recall mathematical expression:
                  Recall = TP/ (TP + FN) --------------------------------(4)
Precision and recall for our models can easily be calculated by inputting the values into the precision and recall equations.
## F1 SCORE
F1 score can be defined as the harmonic mean between the recall and the precision. F1 score can be used for statistical measure used in rating performance.
Mathematical equation for F1 score:
Fbeta = (1 + β2) precision × recall --------------------(5)
                                           β2 × precision + recall 
## ACCURACY
Accuracy is one of the evaluating metrics which measures the number of observations for both positive and negative, that were classified correctly.
Accuracy equation:
Accuracy =   tp + tn -----------------------------(6)
                                              tp + fp + tn + fn

![xgb](https://github.com/user-attachments/assets/7a466455-bd23-44de-8e8c-c149f7e095a2)

![rf](https://github.com/user-attachments/assets/06e836a1-1f82-44a2-8ea6-e068520a9ca5)

![perceptron](https://github.com/user-attachments/assets/411a427a-5824-482b-a07d-191d4a354ba9)

![nb](https://github.com/user-attachments/assets/3748227d-8741-485a-be1f-6b4787fed91f)

![logistic R](https://github.com/user-attachments/assets/276a161b-6ef0-414f-86ef-175b88082a35)

![knn](https://github.com/user-attachments/assets/1a83c3fc-89bb-4928-acae-e5ba78e59ad1)

![ensemble stacking](https://github.com/user-attachments/assets/ea73847e-af0f-46cd-b4db-8f58f20c6968)

![decision tree](https://github.com/user-attachments/assets/5186c567-f0a2-4681-936a-6b4475dadb64)

## ROC AUC
AUC simply means area under the curve. ROC curve must be defined first. ROC visualises the trade-off between true positive rate (TPR), and false positive rate. The receiver operating characteristics (ROC) is one of the best ways of visualising model classifier performance to select the best and suitable operating point, or decision threshold. It can also be used for cross validation of the classifier’s overall performance (Bradley 1997).

## SOFTWARE DEVELOPMENT
A web application was implemented for the purpose of making the heart disease predicting system easy to use and friendly. The graphical user interface was created from the python-based libraries called stream lit. Stream lit is an open-source application framework from python language, which is use for the creation of web applications in data science and for machine learning project in a very short time. Stream lit works well with the major python libraries such as numpy, scikit-learn, matplotlib, pandas etc. 
The web application (GUI) has been developed with twelve features which are displayed on the graphical user interface for service users to enter the information. The values of the features are to be entered based on users’ medical information for diagnostic purpose. The web application developed has been trained by model saved in the job lib, the voting ensembled classification algorithm has been trained to predict user input.
The web application contains the predicting system name, the user input features lists and the predicting phase. The web application is shown below.

![prediction](https://github.com/user-attachments/assets/71595566-f4a0-4b5b-a0bd-f594471370b1)

## DISCUSSION
In the proposed research work, the ensembled technique was adopted based on performances evaluation. Six machine learning algorithms (Logistic Regression, Naïve Bayes Classifier, K- Nearest neighbour Classifier, Random Forest Classifier, Decision Tree Classifier and Extreme Gradient Boosting Classifier) and one deep learning algorithm (Multilayer Perceptron) were implemented to compare individual performance in the classification problem to select the best three to be ensembled. The pre-processed dataset was used in carrying out the experiments for the two approaches and the above listed algorithms for classification were applied. 
## MODEL TUNING
This is the process by which the performance of model is being optimised by setting hyperparameters for the model. The values are basically used to control the learning process of algorithms to find the best performing algorithm when evaluating based on validation set (Feurer and Hutter 2018). 
## Hyperparameters 
The parameters for each model were implemented through random search and manual search with 10 folds cv, the parameters for the three models for ensemble learning are stated below.

Random Forest Classifier: n-estimators= 20, random state= 2, max depth= 5
Decision Trees Classifier: max depth= 6, random state= 1.
Extreme Gradient Boosting Classifier: base score= 0.5, booster= ‘gbtree’, colsample-bylevel=1, colsample-bynode=1, colsample-bytree=0.4, gamma= 0.1, learning rate=0.1, max depth= 10, n-estimators=100, n-jobs=0, random state= 0, reg lamba=1, subsample=1, validate parameters=1.

Three best algorithms (Random Forest Classifier, Decision Tree Classifier and Extreme Gradient Boosting Classifier) were adopted to be ensembled based on performance evaluation in tables 4.1, 4.2, 4.3, and 4.4, while other algorithms had a promising result as well but only the best three were ensembled. Two approaches were developed for implementation of models to compare their performances. The first approach was implemented using all the 14 features in the dataset, while the second approach was implemented using 8 selected features based on feature importance to see if there would be variations in the results and performances based on numbers of features. The two approaches had no significant differences in the yielded general results for the ensembled techniques, but there were slight changes in the results of individual algorithms due to dimensionality reduction. But ensembled approach compensated for the performances (Dietterich 2002).Therefore any of the ensembled methods can be used for prediction and development of web application. The hyperparameters for the algorithms were optimised for effective performance of the models using manual search and random search respectively (Feurer and Hutter 2018).
The outliers were removed, and all 14 features were fit, which shows a good and promising results. Random Forest, Decision Trees, and Extreme Gradient Boosting Classifier had a good evaluating performance. The proposed ensemble classifiers came out with an outstanding result based on the model evaluation metrics (Hossin and Sulaiman 2015).
## ROC
The image below shows the receiver operation characteristic curve (ROC) displaying the performance of each model. Extreme Gradient Boosting Classifier had the highest performing accuracy in the ROC, followed by Random Forest Classifier and Decision Tree Classifier, they are the best three best models as indicated in the ROC. The ROC shows the threshold line in broken green lines, any model that performs below the threshold indicates poor performance (Bradley 1997).
Looking into the confusion matrix of all the models, we would see that the confusion matrix of the proposed model (ensemble classifier) came out with outstanding results, high rate of true positive and true negative, no record of false negative and three record of false positive which looks promising for the model performances (Chicco and Jurman 2020).

![roc curve](https://github.com/user-attachments/assets/722bd5d2-b7dc-4399-8504-d483e01291ba)

## PERFORMANCE COMPARISON AGAINST BENCHMARK STUDIES
Comparing the performance of this study with benchmark studies models proposed by Kavitha et al. 2021, (M. Kavitha et al. 2021), and  (A. Lakshmanarao, A. Srisaila and T. S. R. Kiran 2021) because they had similar study on heart disease prediction using the ensemble technique.
(M. Kavitha et al. 2021) had a research study on heart disease prediction using ensemble, in which a novel method of machine learning was developed. Three machine learning algorithms were implemented which were, Random Forest, Decision Tree, and Hybrid model. Random Forest and Decision Trees were ensembled and the accuracy was 88.7%. Decision Tree was 79% accuracy, while Random Forest yielded 81% accuracy. The model was evaluated using MSE, MAE, RMSE and accuracy.
(D. R. Krithika and K. Rohini 2021) researched on predicting heart disease with ensemble technique. The algorithms implemented were XGBoost with 74% accuracy, Decision Tree with 72% accuracy, KNN (71%), SVM (72%), Logistic Regression (70%), Naïve Bayes (58%), Random Forest (73%), ANN (71%) and Hyperparameter tuned Random Forest Classifier (96%).
(A. Lakshmanarao, A. Srisaila and T. S. R. Kiran 2021) in his study on heart disease prediction using ensemble learning approached his study was through feature selection using ANOVA F-value and Mutual information. The best features were selected using the selection approach, and three techniques were applied which are random over sampling, synthetic minority oversampling and adaptive synthetic sampling approach. Two datasets were used for implementation, one from UCL (Dataset 1) and the other from Kaggle (Dataset 2).
## LIMITATIONS
The time frame for the execution of the thesis was a short one, and due to the limited time caped for the completion of the research work, there were several limitations to put into consideration.
-	Sample size of dataset: The dataset sample size was 1025 with 14 features, which is not large enough to implement for deep learning algorithms and as a result we could only implement just one deep learning algorithm and the result was not impressive but promising.
-	The main participants record in the dataset were from the US, UK, Hungarian and Switzerland, hence it is not a global represented dataset which means the outcome of the result might not favour other races because of genetic variation. 
-	Model evaluation: Due to limited time allocated for the completion of the thesis, we will limit the success metrics of our model evaluation to accuracy, precision, F1 score, ROC, recall and implementing a cross validation to generalise model performance.
-	Dataset features: The dataset used only identifies 14 features as risk factors for cardiovascular disease whereas there are more important risk factors such as excessive alcohol intake, smoking, obesity, unhealthy diet, family history etc that should be considered.
## CONCLUSION AND FUTURE WORK
An ensemble classifier model was proposed in this thesis for heart disease prediction, in which the data set used was a combination of four repository database (Cleveland, Switzerland, Long Beach V and Hungary). Two approaches were implemented, first approach was implemented using 14 features while feature selection was adopted in the second approach and eight features were selected based on feature importance. In the process, various classifier algorithms were used to compare the performances of individual algorithms in which Extreme Gradient Boosting classifier had the best performance with 98% accuracy, then Decision Trees Classifier with 94% accuracy and Random Forest Classifier with 92% accuracy, before ensemble classifier was implemented using stacking classifier and voting classifier, which gave 99% accuracy. The experimental results for the two techniques show that the proposed model was efficient and had a good performance over the benchmark study. 
In the future, the ensemble technique will be applied using large dataset and more of the predictions will be done using deep learning algorithms. Also, the proposed algorithms will be implemented for other dataset with different features.





