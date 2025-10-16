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

<b>Exited</b> is the dependent variable, where a value of <b>1</b> represents <b>customers who have left (churned)</b> and <b>0</b> represents <b>customers who have stayed with the bank</b>.

The dataset was clean, with no missing or duplicate values. I just renamed the <i>Geography</i> feature to <i>Country</i> and removed only two outliers (14 and 19) in the <i>Number_of_Children</i> attribute, as I assessed them as data entry errors.

## Language & Libraries

<b>Language</b>: Python 3<br><br>
<b>Libraries</b>: 
<ul><li><b>Pandas</b> and <b>Numpy</b> for data manipulation</li>
<li><b>Matplotlib</b> and <b>Seaborn</b></li> for visualization</li>
<li>scikit-learn for machine learning (Decision Trees, Logistic Regression, Gradient Boosting, KMeans and SMOTE)</li></ul>

## Results

### Customer Profiles

- <b>Profile 1: women from the Netherlands aged between 40 and 59</b><br>
<b>Attrition rate: 58.81%</b> (257 out of 437).

I performed univariate, bivariate, and multivariate analyses to explore variable distributions and relationships. Using visualization techniques such as <b>Pairplots</b>, I observed a high attrition rate among customers aged 40–59. Further analysis with bar charts and pivot tables confirmed that the combination of <i>Age</i>, <i>Country</i>, and <i>Gender</i> was most significant, revealing that women aged 40–59 from the Netherlands had an attrition rate of 58.81% (257 of 437 customers).

To confirm these results, I turned categorical variables into numbers using the <b>One Hot Encoding (get_dummies)</b> method and ran a <b>Decision Tree Classifier</b> with a 20% test split. The model reached 82% accuracy, but that alone didn’t tell the whole story — recall and precision for the <i>Exited: Yes</i> class were much lower (36% and 53%), showing the model struggled to spot customers who actually left.

After tuning hyperparameters (like max depth and minimum samples per split) without much improvement, I realized the main issue was class imbalance. I fixed this by oversampling the minority class <i>Exited</i> using the <b>SMOTE technique</b> and fine-tuning the model again. This time, performance improved a lot — precision reached 87% and recall 76%.

<b>Feature importance analysis</b> confirmed that <i>Age</i> (1st, 32%), <i>Country_Netherlands</i> (3rd, 11%), and <i>Gender_Female</i> (4th, 10%) were still among the top predictors, backing up the profile I’d already found.
<br><br>

- <b>Profile 2: customers aged between 40 and 59 with a credit score rate between 350 and 559</b><br> 
<b>Attrition rate: 46.54%</b> (229 out of 493).

My first thought was that the second profile would also revolve around the dataset’s most important feature: <i>Age</i>. It was already clear that customers over 40 had higher attrition rates. To explore this further, I looked at how <i>Age</i> related to other features using different visualization techniques. Once again, the <b>Pairplot</b> provided the first clue — it showed a group of <i>Exited</i> customers aged between 40 and 59 with noticeably low credit scores.

To check this pattern, I used the <b>KMeans clustering technique</b> with <i>Age</i> and <i>Credit_Score</i> as inputs. After testing up to ten clusters, the “elbow method” showed that four clusters worked best. When I looked at cluster no. 2, I found that the <i>Credit_Score</i> values ranged from 350 to 559.

Next, I ran a bivariate analysis with a <b>pivot table</b>, combining the four clusters with the <i>Age_Range</i> feature I had used earlier. By calculating the mean, count, and sum of <i>Exited</i> customers, I confirmed that cluster no. 2 — customers aged 40–59 — had the highest attrition rate: 46.54% (229 out of 493).


### Age as Credit Card Eligibility Predictor

After seeing how important <i>Age</i> was in the profile problem, I wanted to check if it also played a key role in predicting credit card eligibility. To do this, I compared <i>Age</i> with other features. Since the visual and correlation analyses didn’t show much, I trained a <b>Decision Tree Classifier</b> on all variables. Using <b>SMOTE</b> to balance the data improved accuracy from 58% to 69%, and the model showed that <i>Credit_Score</i> and <i>Account_Length</i> were stronger predictors than <i>Age</i>. This also matched what’s generally seen in the banking industry.

Next, I ran a <b>Logistic Regression model</b> to see which features had the most influence. Using <i>Credit_Card</i> as the target, the results confirmed that <i>Credit_Score</i>, <i>Age</i>, and <i>Account_Length</i> were the top predictors, with precision scores of 71%, 70%, and 69%.

To dig deeper, I used a <b>Gradient Boosting Classifier</b> to capture more complex relationships. Including extra features like <i>Balance</i>, <i>Years_Employed</i>, and <i>Total_Income</i> increased accuracy to 69.46%, with <i>Balance</i> and <i>Credit_Score</i> ranking highest. When I tested the model again with only the three main features, <i>Credit_Score</i> stayed on top, achieving 69.36% accuracy.

Across all tests, <i>Credit_Score</i> consistently came out as the best predictor of credit card eligibility, clearly outperforming <i>Age</i> and <i>Account_Length</i>.
