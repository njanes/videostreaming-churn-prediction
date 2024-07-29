##### Thursday the 25th of July, 2024
# Predicting Churn Risk in Customers of a Video Streaming Service*

###### *This project is based on [Coursera's data science churn prediction challenge](https://www.coursera.org/projects/data-science-challenge)

---

## Introduction
### Background

Subscription services are leveraged by companies across many industries, from fitness to video streaming to retail. One of the primary objectives of companies with subscription services is to decrease churn and ensure that users are retained as subscribers as it is more costly to procure new customers than to retain exsisting ones. 

Subscription cancellation can happen for a multitude of reasons, including:
* the customer completes all content they were interested in, and no longer need the subscription
* the customer finds themselves to be too busy and cancels their subscription until a later time
* the customer determines that the streaming service is not the best fit for them, so they cancel and look for something better suited

Regardless the reason for cancelling a subscription, this video streaming company has a vested interest in understanding the likelihood of each individual customer to churn in their subscription so that resources can be allocated appropriately to support customers. In order to do this efficiently and systematically, companies employ machine learning to predict which users are at the highest risk of churn.

### Objective

 The goal of this project is to develop a reliable machine learning model that can predict which existing subscribers will continue their subscriptions for another month, so that proper interventions can be effectively deployed to the right audience. In other words, using the patterns found in the `train.csv` data, we will predict whether the subscriptions in `test.csv` will be continued for another month, or not.

----


## Dataset description
The dataset used in this project is a sample of subscriptions initiated in 2021, all snapshotted at a particular date before the subscription was cancelled. The data is split 70/30 into a training set and a test set, which contain 243,787 and 104,480 entries, respectively.

Both `train.csv` and `test.csv` contain one row per unique subscription. For each subscription, a (`CustomerID`) identifier column is included. In addition to this identifier column, the `train.csv` dataset also contains the target variable, a binary column `Churn`.

Both datasets have an identical set of features. Descriptions of each feature are shown below.


```python
import pandas as pd

data_descriptions = pd.read_csv("data_descriptions.csv")
pd.set_option("display.max_colwidth", None)
data_descriptions
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column_name</th>
      <th>Column_type</th>
      <th>Data_type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AccountAge</td>
      <td>Feature</td>
      <td>integer</td>
      <td>The age of the user's account in months.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MonthlyCharges</td>
      <td>Feature</td>
      <td>float</td>
      <td>The amount charged to the user on a monthly basis.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TotalCharges</td>
      <td>Feature</td>
      <td>float</td>
      <td>The total charges incurred by the user over the account's lifetime.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SubscriptionType</td>
      <td>Feature</td>
      <td>object</td>
      <td>The type of subscription chosen by the user (Basic, Standard, or Premium).</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PaymentMethod</td>
      <td>Feature</td>
      <td>string</td>
      <td>The method of payment used by the user.</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PaperlessBilling</td>
      <td>Feature</td>
      <td>string</td>
      <td>Indicates whether the user has opted for paperless billing (Yes or No).</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ContentType</td>
      <td>Feature</td>
      <td>string</td>
      <td>The type of content preferred by the user (Movies, TV Shows, or Both).</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MultiDeviceAccess</td>
      <td>Feature</td>
      <td>string</td>
      <td>Indicates whether the user has access to the service on multiple devices (Yes or No).</td>
    </tr>
    <tr>
      <th>8</th>
      <td>DeviceRegistered</td>
      <td>Feature</td>
      <td>string</td>
      <td>The type of device registered by the user (TV, Mobile, Tablet, or Computer).</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ViewingHoursPerWeek</td>
      <td>Feature</td>
      <td>float</td>
      <td>The number of hours the user spends watching content per week.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AverageViewingDuration</td>
      <td>Feature</td>
      <td>float</td>
      <td>The average duration of each viewing session in minutes.</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ContentDownloadsPerMonth</td>
      <td>Feature</td>
      <td>integer</td>
      <td>The number of content downloads by the user per month.</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GenrePreference</td>
      <td>Feature</td>
      <td>string</td>
      <td>The preferred genre of content chosen by the user.</td>
    </tr>
    <tr>
      <th>13</th>
      <td>UserRating</td>
      <td>Feature</td>
      <td>float</td>
      <td>The user's rating for the service on a scale of 1 to 5.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SupportTicketsPerMonth</td>
      <td>Feature</td>
      <td>integer</td>
      <td>The number of support tickets raised by the user per month.</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Gender</td>
      <td>Feature</td>
      <td>string</td>
      <td>The gender of the user (Male or Female).</td>
    </tr>
    <tr>
      <th>16</th>
      <td>WatchlistSize</td>
      <td>Feature</td>
      <td>float</td>
      <td>The number of items in the user's watchlist.</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ParentalControl</td>
      <td>Feature</td>
      <td>string</td>
      <td>Indicates whether parental control is enabled for the user (Yes or No).</td>
    </tr>
    <tr>
      <th>18</th>
      <td>SubtitlesEnabled</td>
      <td>Feature</td>
      <td>string</td>
      <td>Indicates whether subtitles are enabled for the user (Yes or No).</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CustomerID</td>
      <td>Identifier</td>
      <td>string</td>
      <td>A unique identifier for each customer.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Churn</td>
      <td>Target</td>
      <td>integer</td>
      <td>The target variable indicating whether a user has churned or not (1 for churned, 0 for not churned).</td>
    </tr>
  </tbody>
</table>
</div>



---
## Import Python Modules

First, we import the libraries/modules that will be used in this project:

- pandas
- numpy
- matplotlib
- seaborn
- Scikit-learn


```python
# Import required packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
```

---
## Load the Data

Let's start by loading the dataset `train.csv` into a dataframe `train_df`, and `test.csv` into a dataframe `test_df`


```python
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
```

Next, we will print the first 5 rows of each data frame


```python
print("Training data frame shape:", train_df.shape)
train_df.head()
```

    Training data frame shape: (243787, 21)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AccountAge</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>SubscriptionType</th>
      <th>PaymentMethod</th>
      <th>PaperlessBilling</th>
      <th>ContentType</th>
      <th>MultiDeviceAccess</th>
      <th>DeviceRegistered</th>
      <th>ViewingHoursPerWeek</th>
      <th>...</th>
      <th>ContentDownloadsPerMonth</th>
      <th>GenrePreference</th>
      <th>UserRating</th>
      <th>SupportTicketsPerMonth</th>
      <th>Gender</th>
      <th>WatchlistSize</th>
      <th>ParentalControl</th>
      <th>SubtitlesEnabled</th>
      <th>CustomerID</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>11.055215</td>
      <td>221.104302</td>
      <td>Premium</td>
      <td>Mailed check</td>
      <td>No</td>
      <td>Both</td>
      <td>No</td>
      <td>Mobile</td>
      <td>36.758104</td>
      <td>...</td>
      <td>10</td>
      <td>Sci-Fi</td>
      <td>2.176498</td>
      <td>4</td>
      <td>Male</td>
      <td>3</td>
      <td>No</td>
      <td>No</td>
      <td>CB6SXPNVZA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>57</td>
      <td>5.175208</td>
      <td>294.986882</td>
      <td>Basic</td>
      <td>Credit card</td>
      <td>Yes</td>
      <td>Movies</td>
      <td>No</td>
      <td>Tablet</td>
      <td>32.450568</td>
      <td>...</td>
      <td>18</td>
      <td>Action</td>
      <td>3.478632</td>
      <td>8</td>
      <td>Male</td>
      <td>23</td>
      <td>No</td>
      <td>Yes</td>
      <td>S7R2G87O09</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>73</td>
      <td>12.106657</td>
      <td>883.785952</td>
      <td>Basic</td>
      <td>Mailed check</td>
      <td>Yes</td>
      <td>Movies</td>
      <td>No</td>
      <td>Computer</td>
      <td>7.395160</td>
      <td>...</td>
      <td>23</td>
      <td>Fantasy</td>
      <td>4.238824</td>
      <td>6</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>EASDC20BDT</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32</td>
      <td>7.263743</td>
      <td>232.439774</td>
      <td>Basic</td>
      <td>Electronic check</td>
      <td>No</td>
      <td>TV Shows</td>
      <td>No</td>
      <td>Tablet</td>
      <td>27.960389</td>
      <td>...</td>
      <td>30</td>
      <td>Drama</td>
      <td>4.276013</td>
      <td>2</td>
      <td>Male</td>
      <td>24</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>NPF69NT69N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>16.953078</td>
      <td>966.325422</td>
      <td>Premium</td>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>TV Shows</td>
      <td>No</td>
      <td>TV</td>
      <td>20.083397</td>
      <td>...</td>
      <td>20</td>
      <td>Comedy</td>
      <td>3.616170</td>
      <td>4</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>4LGYPK7VOL</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
print("Test data frame shape:", test_df.shape)
test_df.head()
```

    Test data frame shape: (104480, 20)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AccountAge</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>SubscriptionType</th>
      <th>PaymentMethod</th>
      <th>PaperlessBilling</th>
      <th>ContentType</th>
      <th>MultiDeviceAccess</th>
      <th>DeviceRegistered</th>
      <th>ViewingHoursPerWeek</th>
      <th>AverageViewingDuration</th>
      <th>ContentDownloadsPerMonth</th>
      <th>GenrePreference</th>
      <th>UserRating</th>
      <th>SupportTicketsPerMonth</th>
      <th>Gender</th>
      <th>WatchlistSize</th>
      <th>ParentalControl</th>
      <th>SubtitlesEnabled</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>17.869374</td>
      <td>679.036195</td>
      <td>Premium</td>
      <td>Mailed check</td>
      <td>No</td>
      <td>TV Shows</td>
      <td>No</td>
      <td>TV</td>
      <td>29.126308</td>
      <td>122.274031</td>
      <td>42</td>
      <td>Comedy</td>
      <td>3.522724</td>
      <td>2</td>
      <td>Male</td>
      <td>23</td>
      <td>No</td>
      <td>No</td>
      <td>O1W6BHP6RM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>77</td>
      <td>9.912854</td>
      <td>763.289768</td>
      <td>Basic</td>
      <td>Electronic check</td>
      <td>Yes</td>
      <td>TV Shows</td>
      <td>No</td>
      <td>TV</td>
      <td>36.873729</td>
      <td>57.093319</td>
      <td>43</td>
      <td>Action</td>
      <td>2.021545</td>
      <td>2</td>
      <td>Female</td>
      <td>22</td>
      <td>Yes</td>
      <td>No</td>
      <td>LFR4X92X8H</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>15.019011</td>
      <td>75.095057</td>
      <td>Standard</td>
      <td>Bank transfer</td>
      <td>No</td>
      <td>TV Shows</td>
      <td>Yes</td>
      <td>Computer</td>
      <td>7.601729</td>
      <td>140.414001</td>
      <td>14</td>
      <td>Sci-Fi</td>
      <td>4.806126</td>
      <td>2</td>
      <td>Female</td>
      <td>22</td>
      <td>No</td>
      <td>Yes</td>
      <td>QM5GBIYODA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>88</td>
      <td>15.357406</td>
      <td>1351.451692</td>
      <td>Standard</td>
      <td>Electronic check</td>
      <td>No</td>
      <td>Both</td>
      <td>Yes</td>
      <td>Tablet</td>
      <td>35.586430</td>
      <td>177.002419</td>
      <td>14</td>
      <td>Comedy</td>
      <td>4.943900</td>
      <td>0</td>
      <td>Female</td>
      <td>23</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>D9RXTK2K9F</td>
    </tr>
    <tr>
      <th>4</th>
      <td>91</td>
      <td>12.406033</td>
      <td>1128.949004</td>
      <td>Standard</td>
      <td>Credit card</td>
      <td>Yes</td>
      <td>TV Shows</td>
      <td>Yes</td>
      <td>Tablet</td>
      <td>23.503651</td>
      <td>70.308376</td>
      <td>6</td>
      <td>Drama</td>
      <td>2.846880</td>
      <td>6</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>ENTCCHR1LR</td>
    </tr>
  </tbody>
</table>
</div>



### Summary Statistics

Now, we will return summary statistics for our training data set

First, the count, mean, standard deviation, minimum, maximum, and 25th, 50th, and 75th percentiles of all numeric variables:


```python
train_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AccountAge</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>ViewingHoursPerWeek</th>
      <th>AverageViewingDuration</th>
      <th>ContentDownloadsPerMonth</th>
      <th>UserRating</th>
      <th>SupportTicketsPerMonth</th>
      <th>WatchlistSize</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>243787.000000</td>
      <td>243787.000000</td>
      <td>243787.000000</td>
      <td>243787.000000</td>
      <td>243787.000000</td>
      <td>243787.000000</td>
      <td>243787.000000</td>
      <td>243787.000000</td>
      <td>243787.000000</td>
      <td>243787.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60.083758</td>
      <td>12.490695</td>
      <td>750.741017</td>
      <td>20.502179</td>
      <td>92.264061</td>
      <td>24.503513</td>
      <td>3.002713</td>
      <td>4.504186</td>
      <td>12.018508</td>
      <td>0.181232</td>
    </tr>
    <tr>
      <th>std</th>
      <td>34.285143</td>
      <td>4.327615</td>
      <td>523.073273</td>
      <td>11.243753</td>
      <td>50.505243</td>
      <td>14.421174</td>
      <td>1.155259</td>
      <td>2.872548</td>
      <td>7.193034</td>
      <td>0.385211</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>4.990062</td>
      <td>4.991154</td>
      <td>1.000065</td>
      <td>5.000547</td>
      <td>0.000000</td>
      <td>1.000007</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>8.738543</td>
      <td>329.147027</td>
      <td>10.763953</td>
      <td>48.382395</td>
      <td>12.000000</td>
      <td>2.000853</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60.000000</td>
      <td>12.495555</td>
      <td>649.878487</td>
      <td>20.523116</td>
      <td>92.249992</td>
      <td>24.000000</td>
      <td>3.002261</td>
      <td>4.000000</td>
      <td>12.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>90.000000</td>
      <td>16.238160</td>
      <td>1089.317362</td>
      <td>30.219396</td>
      <td>135.908048</td>
      <td>37.000000</td>
      <td>4.002157</td>
      <td>7.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>119.000000</td>
      <td>19.989957</td>
      <td>2378.723844</td>
      <td>39.999723</td>
      <td>179.999275</td>
      <td>49.000000</td>
      <td>4.999989</td>
      <td>9.000000</td>
      <td>24.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Now, the count, unique count, top class, and frequency of top class for all categorical variables:


```python
train_df.describe(include = 'object')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SubscriptionType</th>
      <th>PaymentMethod</th>
      <th>PaperlessBilling</th>
      <th>ContentType</th>
      <th>MultiDeviceAccess</th>
      <th>DeviceRegistered</th>
      <th>GenrePreference</th>
      <th>Gender</th>
      <th>ParentalControl</th>
      <th>SubtitlesEnabled</th>
      <th>CustomerID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
      <td>243787</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>243787</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Standard</td>
      <td>Electronic check</td>
      <td>No</td>
      <td>Both</td>
      <td>No</td>
      <td>Computer</td>
      <td>Comedy</td>
      <td>Female</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>CB6SXPNVZA</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>81920</td>
      <td>61313</td>
      <td>121980</td>
      <td>81737</td>
      <td>122035</td>
      <td>61147</td>
      <td>49060</td>
      <td>121930</td>
      <td>122085</td>
      <td>122180</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's look at the column names and data type of each column in our data frame


```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 243787 entries, 0 to 243786
    Data columns (total 21 columns):
     #   Column                    Non-Null Count   Dtype  
    ---  ------                    --------------   -----  
     0   AccountAge                243787 non-null  int64  
     1   MonthlyCharges            243787 non-null  float64
     2   TotalCharges              243787 non-null  float64
     3   SubscriptionType          243787 non-null  object 
     4   PaymentMethod             243787 non-null  object 
     5   PaperlessBilling          243787 non-null  object 
     6   ContentType               243787 non-null  object 
     7   MultiDeviceAccess         243787 non-null  object 
     8   DeviceRegistered          243787 non-null  object 
     9   ViewingHoursPerWeek       243787 non-null  float64
     10  AverageViewingDuration    243787 non-null  float64
     11  ContentDownloadsPerMonth  243787 non-null  int64  
     12  GenrePreference           243787 non-null  object 
     13  UserRating                243787 non-null  float64
     14  SupportTicketsPerMonth    243787 non-null  int64  
     15  Gender                    243787 non-null  object 
     16  WatchlistSize             243787 non-null  int64  
     17  ParentalControl           243787 non-null  object 
     18  SubtitlesEnabled          243787 non-null  object 
     19  CustomerID                243787 non-null  object 
     20  Churn                     243787 non-null  int64  
    dtypes: float64(5), int64(5), object(11)
    memory usage: 39.1+ MB
    

Lastly, let's verify there are no missing values in our data:


```python
train_df.isna().any()
```




    AccountAge                  False
    MonthlyCharges              False
    TotalCharges                False
    SubscriptionType            False
    PaymentMethod               False
    PaperlessBilling            False
    ContentType                 False
    MultiDeviceAccess           False
    DeviceRegistered            False
    ViewingHoursPerWeek         False
    AverageViewingDuration      False
    ContentDownloadsPerMonth    False
    GenrePreference             False
    UserRating                  False
    SupportTicketsPerMonth      False
    Gender                      False
    WatchlistSize               False
    ParentalControl             False
    SubtitlesEnabled            False
    CustomerID                  False
    Churn                       False
    dtype: bool



---
## Exploratory Data Analysis

### Distributions of Numeric Features
To visualize the distribution of each numeric feature, we plot a histogram matrix
#### Plotting Histogram Matrix


```python
# Creating numeric feature data frame and list
numeric_features = train_df.select_dtypes(["integer", "float"])
numeric_features_list = list(numeric_features)
numeric_features_index = [train_df.columns.get_loc(c) for c in numeric_features_list if c in train_df]
numeric_features_index
# Creating histogram grid
numeric_features.hist(figsize=(20, 12), layout=(3, 5))
plt.suptitle("Numeric Feature Histogram Matrix", fontsize = 18, fontweight = "bold", y = 0.95)
plt.show()
```


    
![png](churn-predict_files/churn-predict_21_0.png)
    


### Distributions of Categorical Features

#### Plotting Countplot Grid

To simplify the following visualizations, we make a list of numeric and categorical features.


```python
# List of numeric features column names
numeric_features = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_features.remove('Churn')

# List of categorical features column names
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('CustomerID') 
```

Now, we create a function to make the countplot grid


```python
# Countplot function
def make_countplot(features_list):
    i = 0
    for n in range(0, 5):
        m = 0
        a = sns.countplot(
            ax=axes[m, n],
            x=features_list[i],
            data=train_df,
            color="blue",
            order=train_df[features_list[i]].value_counts().index,
        )
        a.tick_params(axis="x", labelrotation=45)
        m += 1
        
        a = sns.countplot(
            ax=axes[m, n],
            x=features_list[i+1],
            data=train_df,
            color="blue",
            order=train_df[features_list[i+1]].value_counts().index,
        )
        a.tick_params(axis="x", labelrotation=45)
        i += 2

# Figure dimensions
fig, axes = plt.subplots(2, 5, figsize=(18, 8))

# Plot categorical countplot matrix
make_countplot(categorical_features)

# Adjust layout for subplot spacing
plt.tight_layout(w_pad=4, h_pad=1)
plt.suptitle("Categorical Feature Countplot Matrix", fontsize = 18, fontweight = "bold", y = 1.05);
```


    
![png](churn-predict_files/churn-predict_26_0.png)
    


### Bivariate Visualization of Categorical Variables
To visualize the relationship between each categorical variable and churn distribution, we plot grouped bar plots depicting both the counts, and percentage of churn in each class of each categorical variable. We do the same separately below for two-class (binary) categorical variables. 

We define a function to facilitate plotting the aforementioned grouped bar plots:


```python
# Creating grouped bar plot function
def grouped_plot(group_col):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    labels = ["Y", "N"]
    grouped = train_df.groupby(group_col)["Churn"].agg(Count="value_counts")
    # Calculate count
    count = grouped.pivot_table(values="Count", index=group_col, columns=["Churn"])
    count.plot(kind="bar", color=["#2e9624", "#eb2d46"], rot=0, ax=ax1)
    # Create count plot
    ax1.set_title("Churn Risk (Count)", fontsize=13, pad=10)
    ax1.set_ylabel("Count", size=10)
    ax1.legend(labels=labels, loc="upper right", title="Renew")
    # Calculate percentage
    perc = grouped.groupby(level=[0]).apply(lambda g: round(g * 100 / g.sum(), 2))
    perc.rename(columns={"Count": "Percentage"}, inplace=True)
    perc = perc.pivot_table(values="Percentage", index=group_col, columns=["Churn"])
    # Create percentage plot
    perc.plot(kind="bar", color=["#2e9624", "#eb2d46"], rot=0, ax=ax2)
    ax2.set_title("Churn Risk (Percentage)", fontsize=13, pad=10)
    ax2.set_ylabel("Percentage", size=10)
    ax2.legend(labels=labels, loc="upper right", title="Renew")

    # Add data labels
    for ax in (ax1, ax2):
        for i in range(0, 2):
            ax.bar_label(ax.containers[i], label_type="edge", fontsize=9)
    plt.suptitle("Churn Distribution by " + str(group_col), fontsize=16, fontweight = "bold")
    plt.tight_layout(rect=[0, 0, 1, 1])
```

#### Visualizing Categorical Feature Churn Distribution


```python
for i in range(0, 10):
    grouped_plot(categorical_features[i])
```


    
![png](churn-predict_files/churn-predict_30_0.png)
    



    
![png](churn-predict_files/churn-predict_30_1.png)
    



    
![png](churn-predict_files/churn-predict_30_2.png)
    



    
![png](churn-predict_files/churn-predict_30_3.png)
    



    
![png](churn-predict_files/churn-predict_30_4.png)
    



    
![png](churn-predict_files/churn-predict_30_5.png)
    



    
![png](churn-predict_files/churn-predict_30_6.png)
    



    
![png](churn-predict_files/churn-predict_30_7.png)
    



    
![png](churn-predict_files/churn-predict_30_8.png)
    



    
![png](churn-predict_files/churn-predict_30_9.png)
    


### Multivariate Visualization of Numeric Variables
#### Correlations
To visualize the interactions between numeric variables (i.e., their correlations), we perform correlation analysis and plot a correlation matrix of the 9 numeric features. 


```python
# Perform correlation analysis for numerical features
correlation_matrix = train_df[numeric_features].corr()

# Plot the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Numerical Feature Correlation Matrix', fontweight = "bold", fontsize = 18, y = 1.05, x = 0.4)
plt.show()
```


    
![png](churn-predict_files/churn-predict_32_0.png)
    


## Feature Importance

Using `RandomForestClassifier()` and `.feature_importances_`, we determine the importance of each of our features to help us decide which we should include in our model.


```python
# Use RandomForestClassifier to find feature importance for numerical and categorical features
rfc = RandomForestClassifier()

# Encode categorical features
encoded_cat = pd.get_dummies(train_df, columns=categorical_features, drop_first=True).drop(['Churn', 'CustomerID'], axis=1)
labels = train_df['Churn']

# Fit the model
rfc.fit(encoded_cat, labels)

# Get feature importances
importances = rfc.feature_importances_

# Creating a DataFrame for feature importances
feature_importance_df = pd.DataFrame({'Feature': encoded_cat.columns, 'Importance': importances})

# Displaying feature importances
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df.head(10)  # Displaying top 10 features
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>AverageViewingDuration</td>
      <td>0.110444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ViewingHoursPerWeek</td>
      <td>0.105446</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MonthlyCharges</td>
      <td>0.097962</td>
    </tr>
    <tr>
      <th>0</th>
      <td>AccountAge</td>
      <td>0.095807</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TotalCharges</td>
      <td>0.095240</td>
    </tr>
    <tr>
      <th>6</th>
      <td>UserRating</td>
      <td>0.086879</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ContentDownloadsPerMonth</td>
      <td>0.082470</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WatchlistSize</td>
      <td>0.062566</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SupportTicketsPerMonth</td>
      <td>0.046271</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MultiDeviceAccess_Yes</td>
      <td>0.013912</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preprocessing
Before building our models, we must convert our features to appropriate formats. We begin by filtering our top features as determined by feature importance above.


```python
# Filter top features
top_features = feature_importance_df.head(10)['Feature'].tolist()
filtered_numeric_columns = [col for col in numeric_features if col in top_features]
filtered_categorical_columns = [col for col in categorical_features if col in top_features]
```

Then, we create transformers for categorical and numeric data, and combine them using `ColumnTransformer()` to create our preprocessor. 


```python
# Create transformers for preprocessing
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, filtered_categorical_columns),
        ('num', numeric_transformer, filtered_numeric_columns)
    ]
)
```

## Model Building
Next, we define Logistic Regression, K-Nearest Neighbours, and Random Forest pipelines to transform the features using our pre-defined preprocessor. 

### Logistic Regression


```python
# Create a logistic regression pipeline with preprocessing
lr_model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', LogisticRegression())])
```

### K Nearest Neigbours


```python
# Create a KNN classifier pipeline with preprocessing
knn_model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', KNeighborsClassifier())])
```

### Random Forest Classifier


```python
random_forest_model = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
)
```

---
## Cross Validation

### Preparing Feature, Target Data

Before we perform cross validation to score our models, we must create separate data frames for the features and target. We also filter the training and test data by our selected features. 


```python
# Selected features
selected_features = [feature for feature in top_features if feature in train_df.columns]

# Splitting training features, target; filtering by selected features
X_train = train_df[selected_features]
y_train = train_df['Churn']

# Filtering test data frame by selected features
X_test = test_df[selected_features]
```


```python
# Cross-validation
lr_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='roc_auc')
knn_scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='roc_auc')
rfc_scores = cross_val_score(random_forest_model, X_train, y_train, cv=5, scoring='roc_auc')

# Calculate the average performance across the folds
lr_performance = lr_scores.mean()
knn_performance = knn_scores.mean()
rfc_performance = rfc_scores.mean()

print("Logistic Regression ROC-AUC:", lr_performance)
print("KNN ROC-AUC:", knn_performance)
print("Random Forest ROC-AUC:", rfc_performance)
```

    Logistic Regression ROC-AUC: 0.7465650106544565
    KNN ROC-AUC: 0.6436138327547012
    Random Forest ROC-AUC: 0.7182908517648123
    

The top performing model based on ROC-AUC score is Logistic Regression, followed by Random Forest, then K-Nearest Neighbours. Based on these metrics, we opt to use the logistic regression model for our predictive analysis. 


---
## Predictions

We now fit our logistic regression model to our training data, then use `.predict_proba()` to predict the churn probability for each `CustomerID`. The predicted values are stored in the newly created `prediction_df` data frame. Lastly, we print the first 10 rows of the data frame to get a quick look at our predicted probabilities.


```python
# Fitting the model
lr_model.fit(X_train, y_train)

# Making predictions on the test set
predicted_probabilities = lr_model.predict_proba(X_test)[:, 1]
predicted_probabilities

# Creating predicted values dataframe
prediction_df = pd.DataFrame({
    'CustomerID': test_df['CustomerID'],
    'predicted_probability': predicted_probabilities
})

# Return first 10 rows of predicted values
print("Predicted probability of churn by CustomerID:")
prediction_df.head(10)
```

    Predicted probability of churn by CustomerID:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>predicted_probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1W6BHP6RM</td>
      <td>0.103661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LFR4X92X8H</td>
      <td>0.039467</td>
    </tr>
    <tr>
      <th>2</th>
      <td>QM5GBIYODA</td>
      <td>0.400630</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D9RXTK2K9F</td>
      <td>0.038432</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENTCCHR1LR</td>
      <td>0.150761</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7A88BB5IO6</td>
      <td>0.392558</td>
    </tr>
    <tr>
      <th>6</th>
      <td>70OMW9XEWR</td>
      <td>0.116946</td>
    </tr>
    <tr>
      <th>7</th>
      <td>EL1RMFMPYL</td>
      <td>0.240051</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4IA2QPT6ZK</td>
      <td>0.186936</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AEDCWHSJDN</td>
      <td>0.181187</td>
    </tr>
  </tbody>
</table>
</div>



## Discussion

The churn probabilities provided by the model developed in this project will allow for the video streaming service to identify "high-risk" customers and thereby allocate it's customer retention resources more efficiently. For example, the company may wish to offer personalized deals or improved service to those at risk of leaving. This may not only result in reduced churn, but also an improved customer experience, which can attract new customers through positive word-of-mouth. Another way in which churn probabilities can ***improve customer experience*** is by allowing for improved customer segmentation, enhancing tailored marketing strategies and product offerings.

From a financial standpoint, the identification and targeting of high-risk customers (churn management) ***increases ROI*** and ***stabilizes revenue***. Accurate churn predictions also aid in financial forecasting and planning, helping make more informed investment decisions, which provides a ***competitive advantage*** in the long run.
