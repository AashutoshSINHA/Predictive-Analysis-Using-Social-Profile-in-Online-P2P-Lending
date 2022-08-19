# Predictive Analysis Using Social Profile in Online P2P Lending 
## 1. Introduction
Prosper is a global market lending platform that has funded over $8 billion in loans. Prosper enables people to invest in one another in a financially and socially rewarding way. Borrowers list loan requests ranging from $2,000 to $35,000 on Prosper, and individual investors can invest as little as $25 in each loan listing they choose. Prosper manages loan servicing on behalf of the matched borrowers and investors. We will clean up the data and perform exploratory data analysis on the loan data in the following sections, using univariate, bivariate, and multivariate graphs and summaries. The analysis section highlights the most intriguing observations from the plots section. In Final Plots and Summary section, we will identify top three charts and provide final reflections regarding the dataset.

Individual consumers can borrow and lend money to one another directly through online peer-to-peer (P2P) lending platforms. In an experiment, we look at the borrowers, loans, and groups that influence performance predictability. by conceptualizing financial and social strength for the online P2P lending industry. forecast the borrower's rate and if the loan will be paid on time as a result of using a database of 9479 completed P2P transactions, we conducted an empirical investigation. Transactions in the calendar year 2007 support the suggested. This study used a conceptual model. The findings revealed that searching for financial files.P2P performance prediction can be improved by using social indicators. The loan market Although social factors influence borrowing rates, when compared to financial strength, the effects of and status are quite minimal.
What we Do: 
|-------------------|
| Tasks: EDA (Exploratory Data Analysis):| 
| 1. Perform Data Exploration. 
| 2. Data Cleaning. 
| 3. Data Visualization and Manipulation. 
| 4. Perform EDA on the file separately.
| 5. Trained the model using machine learning algorithm 
 
## Understanding the Dataset
The dataset under consideration contains information from loans taken out between 2005 and 2014. There are variables related to the borrower's loan history, credit history, occupation, income range, and so on. 
A few of the dataset's columns are listed below:
|-----------------------------------| 
| ListingNumber                     |  
| ListingCreationDate               | 
| CreditGrade                       |
| Term                              |
| LoanStatus                        |
| ClosedDate                        |
| BorrowerAPR                       |
| BorrowerRate                      |
| LenderYield                       |
| EstimatedEffectiveYield           |    
| EstimatedLoss                     |
| EstimatedReturn                   |
| ProsperRating                     |
![image](https://user-images.githubusercontent.com/88158022/185349591-6c8e6d7f-775a-4b6c-a4a6-dcf6c9d52c09.png)


# 2. Exploratory data Analysis (EDA)
## 2.1 Data Wraglling or Data Cleaning
Let's start with the Introduction "According to the [ techtarget.com](https://www.techtarget.com/searchdatamanagement/definition/data-scrubbing) the data cleaning or data Wraglling or data scrubbing, is the process of fixing incorrect, incomplete, duplicate or otherwise erroneous data in a data set. It involves identifying data errors and then changing, updating or removing data to correct them. Data cleansing improves data quality and helps provide more accurate, consistent and reliable information for decision-making in an organization. 
A few of the dataset's columns are listed below:
|-----------------------------------| 
| ListingNumber                     |  
| ListingCreationDate               | 
| CreditGrade                       |
| Term                              |
| LoanStatus                        |
| ClosedDate                        |
| BorrowerAPR                       |
| BorrowerRate                      |
| LenderYield                       |
| EstimatedEffectiveYield           |    
| EstimatedLoss                     |
| EstimatedReturn                   |
| ProsperRating                     |

This data table stored as 'df' has 113937 rows and 81 columns. Following that, because we have 81 variables and some of the cells may have missing data, I will remove the columns with more than 80% NA's.
Cleaning the dataset 

- **Our dataset** contains null values mainly in the form of "?" however because pandas cannot read these values, we will convert them to np.nan form.
```
df.duplicated().sum()
```
```
for col in df.columns:
    df[col].replace({'?':np.NaN},inplace=True)
```
 
*Now, that the data appears to be clean, let's plot some of the most important plotted graphs.* 

## Univariate Plots Section ##

A univariate plot depicts and summarizes the data's distribution. Individual observations are displayed on a dot plot, also known as a strip plot. A box plot depicts the data in five numbers: minimum, first quartile, median, third quartile, and maximum.

***Research Question 1*** : What are the most number of borrowers Credit Grade? 

- *Check the univariate relationship of Credit Grade*

```
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.countplot(y='CreditGrade',data=df)
```
![image](https://user-images.githubusercontent.com/88158022/185348116-4d5e20ca-138b-48bc-9b66-3a2c214b8c7b.png)

***Research Question 2*** : Since there are so much low Credit Grade such as C and D , does it lead to a higher amount of deliquency?

- *Check the univariate relationship of Loan Status*

```
df['LoanStatus'].hist(bins=100)

```
![image](https://user-images.githubusercontent.com/88158022/185348239-4b4de844-fced-4903-93d8-e441f0eb218b.png)

***Research Question 3*** : What is the highest number of BorrowerRate?

- *Check the univariate relationship of  Borrower rate*

```
df['BorrowerRate'].hist(bins=100)

```
![image](https://user-images.githubusercontent.com/88158022/185348335-0e4f032d-88cd-41dd-bcc8-4c1870f5f8bd.png)

***Research Question 4*** :  Since the highest number of Borrower Rate is between 0.1 and 0.2, does the highest number of Lender Yield is between 0.1 and 0.2?

- *Check the univariate relationship of Lender Yield on Loan*

```
 df['LenderYield'].hist(bins=100)
```
![image](https://user-images.githubusercontent.com/88158022/185348470-e6721a17-301d-4d95-9ac4-08dc127c361f.png)

## Bivariate Plots Section ##

Bivariate analysis is a type of statistical analysis in which two variables are compared to one another. One variable will be dependent, while the other will be independent. X and Y represent the variables. The differences between the two variables are analyzed to determine the extent to which the change has occurred.

*Discuss some of the relationships you discovered during this phase of the investigation. What differences did the feature(s) of interest have with other features in the dataset?*

My main point of interest was the borrower rate, which had a strong correlation with the Prosper Rating. Borrower rate increased linearly as rating decreased.Listed below are some good example of research questions that have been subjected to bivariate analysis.

***Research Question 1*** : Is the Credit Grade really accurate? Does higher Credit Grade leads to higher Monthly Loan Payment? As for Higher Credit Grade we mean from Grade AA to B.

- *Check the Bivariate Relationship between CreditGarde and MonthlyLoan Payment.*

```
base_color = sns.color_palette()[3]
plt.figure(figsize = [20, 5])
plt.subplot(1, 2, 2)
sns.boxplot(data=df,x='CreditGrade',y='MonthlyLoanPayment',color=base_color);
plt.xlabel('CreditGrade');
plt.ylabel('Monthly Loan Payment');
plt.title(' Relationship between Creditgrade and MonthlyLoan Payment');
```
![image](https://user-images.githubusercontent.com/88158022/185348605-ec930b08-ee23-41e6-9f5a-2f33e1726a2b.png)

***Research Question 2*** : Here we look at the Completed Loan Status and Defaulted Rate to determine the accuracy of Credit Grade.

- *Check the Bivariate Relatonship between CreditGrade and LoanStatus*

```
base_color = sns.color_palette()[3]
plt.figure(figsize = [20, 5])
plt.subplot(1, 2, 2)
sns.boxplot(data=df,x='CreditGrade',y='MonthlyLoanPayment',color=base_color);
plt.xlabel('CreditGrade');
plt.ylabel('Monthly Loan Payment');
plt.title(' Relationship between Creditgrade and MonthlyLoan Payment');
```
## Multivariate Plots ##
Multivariate analysis is traditionally defined as the statistical study of experiments in which multiple measurements are taken on each experimental unit and the relationship between multivariate measurements and their structure is critical for understanding the experiment.
So, let's look at an example of a research question that we solved using multivariate plots and matplotlib.

***Research Question 1*** : Now we know the Credit Grade is accurate and is a tool that is used by the organization in determining the person’s creditworthiness. Now we need to understand does the ProsperScore, the custom built risk assesment system is being used in determing borrower’s rate?

- *Check the Multivariate Relationship between BorrowerRate and BorrowerAPR.*

```
plt.figure(figsize = [20, 5])
plt.subplot(1, 2, 1)
plt.scatter(data=df,x='BorrowerRate',y='BorrowerAPR',color=base_color);
plt.xlabel('Borrower Rate');
plt.ylabel('Borrower APR');
plt.title('Relationship Between Borrower Rate and BorrowerAPR');

```
![image](https://user-images.githubusercontent.com/88158022/185348772-c1fb3040-9dfc-4ceb-ae8a-1b450915c3b1.png)

```
g = sb.FacetGrid(data = df, col = 'Term', height = 5,
                margin_titles = True)
g.map(plt.scatter, 'BorrowerRate', 'BorrowerAPR');
plt.colorbar()
```
![image](https://user-images.githubusercontent.com/88158022/185348841-d9b04b27-1810-4604-a12c-d8f05bd6dded.png)

**From** a theoretical standpoint, if the higher ProsperScore leads to lower Borrower Rate and Borrower Annual Percentage Rate that means the Prosper Score is being used alongside the Credit Grade in determing a person’s creditworthiness.

# 3. Development Process
We have been tasked with developing a machine learning model using Logistic Regression, Regularized Logistic Regression, Naive Bayes, and the decision tree algorithm. After training the model, we must create a web application with Django or Flask but we have built in using Streamlit. Using the trained machine learning algorithm, the web application can predict whether we have to give the prosper loan yes or no. The task is divided into two groups, Team-A and Team-B, and we have chosen the **scrum methodology** to develop the model.

**Scrum** encourages teams, like a rugby squad, to learn via experience, self-organize while working on a problem, and reflect
on their triumphs and defeats in order to constantly improve (from which it gets its name).

## 3.1 Scrum Methodology Phases and Process:
Scrum procedures focus on the activities and flow of a Scrum project. SBOK® Guide has 19
procedures in all, which are organized into five phases:
- Initiate
- Plan and Estimate
- Implement
- Plan and Estimate
- Release

# 4. Heatmap
There is positive correlation between variable Loan number and lisiting number.There is positive correlation between variable employment status duration total traders, availble bank card credit. There is negative relationship between borrower rate and borrower APR.

# 5. Confusion Matrix
    Reduce the confusion in the Confusion Matrix.
**A confusion matrix** is a technique for summarizing a classification algorithm's performance.
![Confusion matrix of Prosper loanData](https://user-images.githubusercontent.com/88158022/185347768-a4151e5c-e871-4972-b869-a5be1fdbe36f.png)

When you have an unequal number of observations in each class or more than two classes in your dataset, classification accuracy alone can be misleading.

# 6. Label Encoding 
By using label encoding technique we convert all the categorical data into numerical data. Then also convert the loan status column into binary data. 

## 7. Splitting data into training and testing data
Split our dataset into training and testing data.

## Feature Selection
For feature selection we used mutual info classifier. And select the most important features. 
 
## Standardizing data
Then standardize the dataset by using StandardScaler().

# 8. Model Building 
 ### Metrics considered for model evaluation ###
 ***Accuracy, Precision, Recall and F1 Score***
-	Accuracy : What proportion of actual positives and negatives is correctly classified? 
-	Precision : What proportion of predicted positives are truly positive? 
-	Recall: What proportion of actual positives is correctly classified?
-	F1 Score: Harmonic mean of Precision and Recall 
## 8.1 Logistic Regression

**what is Logistic Regression?**

*Well, according to Ian Goodfellow*

“Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.”
- Logistic regression helps to find how probabilities are changed with actions.
- It not only provides a measure of how appropriate a predictor(coefficient size)is, but also its direction of association (positive or negative).
- Logistic regression is less inclined to over-fitting but it can overfit in high dimensional datasets.One may consider Regularization (L1 and L2) techniques to avoid over-fittingin these scenarios.

now, Applying the logistic Regression algorithm to train the dataset of the Prosper P2P lending market.

```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
# training the model
clf.fit(X_train_std, Y_train)
y_pred = clf.predict(X_test_std)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
```
*Evaluating the Model*
```
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test,y_pred))
confusion_matrix(Y_test,y_pred)
(14281+6794)/(14281+898+815+6794)

Output: 0.9248288572933123
```
**The Accuracy of the Model Was 92%.**

# 8.2 Regularized Logistic Regression
Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.” In other words: regularization can be used to train models that generalize better on unseen data, by preventing the algorithm from overfitting the training dataset.

Implement of regularization Logistic regression into the dataset are:

```
import numpy as np
mul = np.matmul

"""
X is the design matrix
y is the target vector
theta is the parameter vector
lamda is the regularization parameter
"""

def sigmoid(X):
    return np.power(1 + np.exp(-X), -1)

"""
hypothesis function
"""
def h(X, theta):
    return sigmoid(mul(X, theta))

"""
regularized cost function
"""
def j(theta, X, y, lamda=None):
    m = X.shape[0]
    theta[0] = 0
    if lamda:
        return (-(1/m) * (mul(y.T, np.log(h(X, theta))) + \
                          mul((1-y).T, np.log(1 - h(X, theta)))) + \
                (lamda/(2*m))*mul(theta.T, theta))[0][0] 
    return -(1/m) * (mul(y.T, np.log(h(X, theta))) + \
                     mul((1-y).T, np.log(1 - h(X, theta))))[0][0]
```
```
"""
regularized cost gradient
"""
def j_prime(theta, X, y, lamda=None):
    m = X.shape[0]
    theta[0] = 0
    if lamda:
        return (1/m) * mul(X.T, (h(X, theta) - y)) + (lamda/m) * theta 
    return (1/m) * mul(X.T, (h(X, theta) - y)) 

"""
Simultaneous update
"""
def update_theta(theta, X, y, lamda=None):
    return theta - alpha * j_prime(theta, X, y, lamda)
```

## 8.3 Naive Bayes
The term "Naïve Bayesian classifiers" refers to a set of classification algorithms based on Bayes' Theorem. It is a family of algorithms that all share a common principle, namely that every pair of features being classified is independent of each other.
Bayes Theorem provides a principled way for calculating the conditional probability.

The simple form of the calculation for Bayes Theorem is as follows:

P(A|B) = P(B|A) * P(A) / P(B)

We can frame classification as a conditional classification problem with Bayes Theorem as follows:

P(yi | x1, x2, …, xn) = P(x1, x2, …, xn | yi) * P(yi) / P(x1, x2, …, xn)

The prior P(yi) is easy to estimate from a dataset, but the conditional probability of the observation based on the class P(x1, x2, …, xn | yi) is not feasible unless the number of examples is extraordinarily large, e.g. large enough to effectively estimate the probability distribution for all different possible combinations of values.

We now went into greater detail about the introduction and how the simple form of the Bayes Theorem calculation is performed. Let's see what happens when we apply the naive bayes algorithm to our own dataset.
```
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_std, Y_train)
y_pred = gnb.predict(X_test_std
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test
```
## 8.4 Decision three
Let's start with the intro "Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome."

**Why too use decision tree algorithm**
Below are the two reasons for using the Decision tree:

- Decision Trees usually mimic human thinking ability while making a decision, so it is easy to understand.
- The logic behind the decision tree can be easily understood because it shows a tree-like structure.

Let's try it on our own dataset and see how accurate it is.

```
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "entropy")

model.fit(X_train_std, Y_train)

y_pred = model.predict(X_test_std)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test,y_pred))
```
**The Accuracy of the Model was 98%**


## 9.  Deployment
### 9.1.1 Streamlit

The development process of the web application was Developed into the Streamlit.
Streamlit is an open source app framework in Python language. 
It helps us create web apps for data science and machine learning in a short time.
It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc. 

**You can access our app by follosing this link [Team-A-predictive-analysis-using-social-profile-streamlitapp.com](https://suwarna93-predictive-analysis-using-social-profile-i-app-f907cz.streamlitapp.com/)**
-	It is a tool that lets you creating applications for your machine learning model by using simple python code. 
-	We write a python code for our app using Streamlit.
-	The output of our app will be Accepted or Rejected. 

                                                 Congratulation🎉

###### You've Successfully Developed you're machine learning Model that can predict the prosper Loan Status ####


**Overview of Web Application:**
![image](https://user-images.githubusercontent.com/88158022/185347659-b8f1e07a-e3b6-496b-9bc6-86bf3c20d0f8.png)

[Visit The Web App:](https://suwarna93-predictive-analysis-using-social-profile-i-app-f907cz.streamlitapp.com/)
