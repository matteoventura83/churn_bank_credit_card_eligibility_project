# Data Mining project for Customer Attrition and Credit Card Eligibility in the Banking Sector

## Overview & Objectives

This project is the result of my assignment for the Data Mining module of my MSc in Computer Science with Data Analytics. It focuses on applying data mining and machine learning techniques to a banking dataset in order to explore customer behavior and credit card eligibility. It addresses two main problem areas:
<ul><li><b>Profiles</b> — identifying and definining <b>two customer segments</b> with a <b>high likelihood of attrition</b></li>
<li><b>Age</b> — evaluateing whether age is a <b>reliable predictor of credit card eligibility</b>, compared to other variables</li></ul>

This work applies statistical analysis, visualization, and machine learning models to extract insights that can support strategic decisions in the banking sector.

## Dataset

The dataset contains anonymized information about bank customers. Each record represents one customer and includes the following attributes:
<ul>
  <li>CustomerID</li>
  <li>Country</li>
  <li>Gender</li>
  <li>Age</li>
  <li>Car_Owner</li>
  <li>Property_Owner</li>
  <li>Credit_Score</li>
  <li>Tenure</li>
  <li>Balance</li>
  <li>Employment_Status</li>
  <li>Years_Employed</li>
  <li>Job_Type</li>
  <li>Education</li>
  <li>Family_Status</li>
  <li>Number_of_Children</li>
  <li>House_Type</li>
  <li>Total_Income</li>
  <li>Account_Length</li>
  <li>Credit_Card</li>
  <li>Exited</li>
</ul>

<b><i>Exited</i></b> is the dependent variable, where a value of <b>1</b> represents <b>customers who have left</b> and <b>0</b> represents <b>customers who have stayed with the bank</b>.

## Language & Libraries

<b>Language</b>: Python 3<br><br>
<b>Libraries</b>: 
<ul><li><b>Pandas</b> and <b>Numpy</b> for data manipulation</li>
<li><b>Matplotlib</b> and <b>Seaborn</b></li> for visualization</li>
<li><b>Scikit-learn</b> for machine learning (Decision Trees, Logistic Regression, Gradient Boosting, KMeans and SMOTE)</li></ul>

## Results

### Customer Profiles

- <b>Profile 1</b>: women from the Netherlands aged between 40 and 59<br>
<b>Attrition rate: 58.81%</b> (257 out of 437).

The dataset was clean, with no missing or duplicate values. I just renamed the <i>Geography</i> feature to <i>Country</i> and removed only two outliers (14 and 19) in the <i>Number_of_Children</i> attribute, as I assessed them as data entry errors.

I performed <b>univariate, bivariate, and multivariate analyses</b> to explore and understand the distribution of the variables. For the numerical features <i>Age</i>, <i>Total_Income</i>, <i>Years_Employed</i>, <i>Credit_Score</i>, and <i>Balance</i>, I even created tailored ranges in order to obtain clearer insights. The <b>Pairplot</b> technique, with the <i>Exited</i> feature serving as the hue, highlighted a high attrition rate in 40-59 age range across all the numerical features. At this point, I was already confident that <i>Age</i> would be the most important feature and that the first profile should have been built around it.

I further explored the correlation between <i>Age</i> and other features with bar charts and pivot tables. In the latter, the aggregate functions <i>mean</i>, <i>count</i> and <i>sum</i> helped me to analyse the number of <i>Exited</i> customers, whose results reinfoirced my initial assumption. In particular, it was the combination of <i>Age</i>, <i>Country</i> and <i>Gender</i> features to provide the most impactful results, revealing that women aged 40–59 from the Netherlands had an attrition rate of 58.81% (257 of 437 customers).

I wanted further confirmation of these outcomes. I turned categorical variables into numbers using the <b>One Hot Encoding</b> method and ran a <b>Decision Tree Classifier</b> with a 20% test split. The model reached 82% accuracy, but recall and precision metrics for the <i>Exited: Yes</i> class were much lower (36% and 53%), indicating the model was struggling to correctly predict and identify positive cases. After unsuccessfully fine-tuning hyperparameters, I suspected the main issue was the dataset's imbalance. To fix this problem, I oversampled the minority class <i>Exited</i> using the <b>SMOTE technique</b>, fine-tuned the hyperparameters again, and the model's performance improved significantly: <i>Exited</i> precision was 87% and recall was 76%. The feature importance analysis confirmed that <i>Age</i> (1st position, 32%), <i>Country_Netherlands</i> (3rd position, 11%), and <i>Gender_Female</i> (4th position, 10%) remained the most influential features in the dataset, confirming the profile I discovered earlier.

<br>

- <b>Profile 2</b>: customers aged between 40 and 59 with a credit score rate between 350 and 559<br> 
<b>Attrition rate: 46.54%</b> (229 out of 493).

I initially hypothesized that the second profile should also be built around what I considered the most significant variable: <i>Age</i>. In particular, it was becoming increasingly evident that the attrition rate was strongly related to people over the age of 40. For this reason, I analyzed again the correlation of <i>Age</i> with other features using various visualization techniques. Pairplot, in particular, helped me identify a potential cluster of <i>Exited</i> customers aged 40–59 with very low credit scores.

To validate this pattern, I used the <b>KMeans clustering technique</b> by selecting the features <i>Age</i> and <i>Credit_Score</i>. The "elbow method" showed that the number of four clusters was the optimal solution, and when I examined all of them, I noticed the minimum and maximum values of cluster n. 2 ranged from 350 to 559. Consequently, I started a bivariate analysis by using the pivot table, combining the four clusters and the <i>Age_Range</i> variable created for the first profile and analysing the number of <i>Exited</i> customers using the aggregate functions <i>mean</i>, <i>count</i> and <i>sum</i>. In cluster n. 2 the age group of 40-59 confirmed the highest attrition rate: 46.54% (229 out of 493).


### Age as Credit Card Eligibility Predictor

After seeing how important <i>Age</i> was in the profile problem, I wanted to check if it also played a key role in predicting credit card eligibility. To do this, I compared <i>Age</i> with other features. Since the visual and correlation analyses didn’t show much, I trained a <b>Decision Tree Classifier</b> on all variables. Using <b>SMOTE</b> to balance the data improved accuracy from 58% to 69%, and the model showed that <i>Credit_Score</i> and <i>Account_Length</i> were stronger predictors than <i>Age</i>. This also matched what’s generally seen in the banking industry.

Next, I ran a <b>Logistic Regression model</b> to see which features had the most influence. Using <i>Credit_Card</i> as the target, the results confirmed that <i>Credit_Score</i>, <i>Age</i>, and <i>Account_Length</i> were the top predictors, with precision scores of 71%, 70%, and 69%.

To dig deeper, I used a <b>Gradient Boosting Classifier</b> to capture more complex relationships. Including extra features like <i>Balance</i>, <i>Years_Employed</i>, and <i>Total_Income</i> increased accuracy to 69.46%, with <i>Balance</i> and <i>Credit_Score</i> ranking highest. When I tested the model again with only the three main features, <i>Credit_Score</i> stayed on top, achieving 69.36% accuracy.

Across all tests, <i>Credit_Score</i> consistently came out as the best predictor of credit card eligibility, clearly outperforming <i>Age</i> and <i>Account_Length</i>.
