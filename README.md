# Data Mining for Customer Attrition and Credit Card Eligibility in the Banking Sector

## Overview & objectives

This project is the result of an assignment for the Data Mining module of my MSc in Computer Science with Data Analytics. It focuses on applying data mining and machine learning techniques to a banking dataset in order to explore customer behavior and credit card eligibility. It addresses two main problem areas:
<ul><li><b>Profiles</b> — identifying and definining <b>two customer segments</b>b> with a <b>high likelihood of attrition</b></li>
<li><b>Age</b> — evaluateing whether age is a <b>reliable predictor of credit card eligibility</b>, compared to other variables</li></ul>

The work applies statistical analysis, visualization, and machine learning models to extract insights that can support strategic decisions in the banking sector.

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

The <b>Exited variable</b> serves as the target label, where a value of <b>1</b> represents <b>customers who have left (churned)</b> and <b>0</b> represents <b>customers who have stayed with the bank</b>.

I initially imported <b>pandas</b>, <b>numpy</b>, <b>seaborn</b>, and <b>matplotlib.pyplot</b> for analysis. The dataset was clean, with no missing or duplicate values, though I renamed the Geography feature to Country and removed only two outliers in the <i>Number_of_Children</i> attribute.

## Methodology & Results

### Customer Profiles

<b>Profile 1: women from the Netherlands aged between 40 and 59</b><br>
Attrition rate: 58.81% (257 out of 437).

I performed univariate, bivariate, and multivariate analyses to explore variable distributions and relationships. Using visualization techniques such as <b>Pairplots</b>, I observed a high attrition rate among customers aged 40–59. Further analysis with bar charts and pivot tables confirmed that the combination of <i>Age</i>, <i>Country</i>, and <i>Gender</i> was most significant, revealing that women aged 40–59 from the Netherlands had an attrition rate of 58.81% (257 of 437 customers).

To solidify these findings, I converted categorical variables into a numerical format using the <b>One Hot Encoding technique get_dummies</b> and I proceeded with the <b>Decision Tree Classifier</b> algorithm with a 20% test size.  Although the accuracy of the model was at 82%, this metric alone was insufficient to evaluate its performance. My focus was more on the recall and precision metrics for the <i>Exited: Yes</i> class, where the values were considerably lower (36% and 53%, respectively), indicating that the model struggled to predict and correctly identify positive cases. After attempting to fine-tune hyperparameters (such as max depth, minimum samples per leaf, and minimum samples per split) without improving performance, I concluded that the main issue was the dataset's significant imbalance. I addressed this problem by oversampling the minority class <i>Exited</i> using the <b>SMOTE technique</b> and fine-tuning again hyperparameters, and the model's performance improved significantly, yielding an <i>Exited</i> precision of 87% and recall of 76%. 

The subsequent feature importance analysis confirmed that <i>Age</i> (1st position, 32%), <i>Country_Netherlands</i> (3rd position, 11%) and <i>Gender_Female</i> (4th position, 10%) remained among the most influential features in the dataset, reinforcing the profile I discovered before.
<br><br>

<b>Profile 2: customers aged between 40 and 59 with a credit score rate between 350 and 559</b><br> 
Attrition rate: 46.54% (229 out of 493).

My first assumption was that also the second profile should primarily be centred around what had now emerged as the dataset's most significant feature: <i>Age</i>. Additionally, it was becoming increasingly apparent that the attrition rate was closely tied to individuals over the age of 40. For this reason, I analysed the correlation of <i>Age</i> with other features using various visualization techniques, and once again, the <b>Pairplot</b> offered the initial insights: I identified a potential cluster of Exited customers, aged between 40 and 59, marked by particularly low credit scores. 

To validate this hypothesis, I employed the <b>KMeans clustering technique</b> by selecting the features <i>Age</i> and <i>Credit_Score</i>. I calculated the inertia for ten different clusters, and the "elbow point" confirmed that four clusters were optimal. Upon examining the minimum and maximum <i>Credit_Score</i> values for cluster n. 2, I found they ranged from 350 to 559.

Consequently, I started a bivariate analysis with the <b>pivot table</b>, combining the four clusters and the same <i>Age_Range</i> created previously for the first profile and analysing the number of Exited customers using the aggregate functions mean, count and sum. In cluster n. 2 the age group of 40-59 confirmed the highest attrition rate: 46.54% (229 out of 493).

### Age as Predictor

After seeing how important <i>Age</i> was in the profile problem, I wanted to check if it also played a key role in predicting credit card eligibility. To do this, I compared <i>Age</i> with other features. Since the visual and correlation analyses didn’t show much, I trained a <b>Decision Tree Classifier</b> on all variables. Using <b>SMOTE</b> to balance the data improved accuracy from 58% to 69%, and the model showed that <i>Credit_Score</i> and <i>Account_Length</i> were stronger predictors than <i>Age</i>. This also matched what’s generally seen in the banking industry.

Next, I ran a <b>Logistic Regression model</b> to see which features had the most influence. Using <i>Credit_Card</i> as the target, the results confirmed that <i>Credit_Score</i>, <i>Age</i>, and <i>Account_Length</i> were the top predictors, with precision scores of 71%, 70%, and 69%.

To dig deeper, I used a <b>Gradient Boosting Classifier</b> to capture more complex relationships. Including extra features like <i>Balance</i>, <i>Years_Employed</i>, and <i>Total_Income</i> increased accuracy to 69.46%, with <i>Balance</i> and <i>Credit_Score</i> ranking highest. When I tested the model again with only the three main features, <i>Credit_Score</i> stayed on top, achieving 69.36% accuracy.

Across all tests, <i>Credit_Score</i> consistently came out as the best predictor of credit card eligibility, clearly outperforming <i>Age</i> and <i>Account_Length</i>.
