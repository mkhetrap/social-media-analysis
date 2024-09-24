import pandas as pd 
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score

df = pd.read_csv('instagram_data.csv')
#print(df.head())

# check for missing values for removal -> no missing values!
#print(df.isnull().sum())

# calculate mean and standard deviation
mean_likes = df['likes'].mean()
std_likes = df['likes'].std()

# Calculate low and high thresholds for likes (z-score calc)
low_threshold = mean_likes - (3 * std_likes)
high_threshold = mean_likes + (3 * std_likes)

# Print the thresholds
# print(f"Low threshold for likes (outlier): {low_threshold}")
# print(f"High threshold for likes (outlier): {high_threshold}")
#Results: 
    #Low threshold for likes (outlier): -397837.2414309727
    #High threshold for likes (outlier): 764344.3753807745


# calculate z-scores to detect instagram likes outliers 
df['zscore_likes'] = np.abs(stats.zscore(df['likes']))
outliers = df[df['zscore_likes'] > 3]
print(f"Number of outliers detected: {outliers.shape[0]}")
#Results: 
    #Number of outliers detected: 67


#example_outlier = outliers.sample(n=1)
#print(example_outlier)


# remove the outliers (was about ~2% of total data)
df_cleaned = df[df['zscore_likes'] <= 3]
df_cleaned = df_cleaned.drop(columns=['zscore_likes'])


#feature manipulations 

#convert timestamp to datetime
df_cleaned['t'] = pd.to_datetime(df_cleaned['t'], unit='s')

#get day of the week and hour of the day
df_cleaned['day_of_week'] = df_cleaned['t'].dt.dayofweek
df_cleaned['hour_of_day'] = df_cleaned['t'].dt.hour

#calculate engagement rate (likes per follower)
df_cleaned['engagement_rate'] = df_cleaned['likes'] / df_cleaned['follower_count_at_t']

#print(df_cleaned.head())

#descriptive_stats = df_cleaned.describe()
#print(descriptive_stats)


#correlation matrix and plots!

# numeric_columns = df_cleaned.select_dtypes(include=[np.number])
# correlation_matrix = numeric_columns.corr()
# print(correlation_matrix)

# #visualize the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# #plotting distribution of likes
# plt.figure(figsize=(12, 6))
# sns.histplot(df_cleaned['likes'], bins=30, kde=True)
# plt.title('Distribution of Likes')
# plt.xlabel('Likes')
# plt.ylabel('Frequency')
# plt.show()

# #plot for Likes vs. Followers
# plt.figure(figsize=(12, 6))
# sns.scatterplot(data=df_cleaned, x='follower_count_at_t', y='likes')
# plt.title('Likes vs. Followers')
# plt.xlabel('Followers')
# plt.ylabel('Likes')
# plt.show()


#Approach 1. Classification! 

# function to classify likes based on mean and standard deviation
def classify_likes(likes):
    mean_likes = 168160  
    std_likes = 154752    

    low_threshold = mean_likes - std_likes  
    high_threshold = mean_likes + std_likes  

    if likes < low_threshold:  #low likes
        return 0
    elif likes < high_threshold:  #medium likes
        return 1
    else:  #high likes
        return 2

df_cleaned['likes_category'] = df_cleaned['likes'].apply(classify_likes)

#classification preperation
X = df_cleaned[['follower_count_at_t', 'day_of_week', 'hour_of_day', 'engagement_rate']]
y_class = df_cleaned['likes_category']
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

#using Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_class, y_train_class)
y_pred_rf = rf_model.predict(X_test_class)

#evaluate the classification model
print("Random Forest Classifier Performance:")
print(f"Accuracy: {accuracy_score(y_test_class, y_pred_rf)}")
print(classification_report(y_test_class, y_pred_rf))



#Approach 2. Regression model! (for actual likes)

y_reg = df_cleaned['likes'] 
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

#initialize linear regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = linear_reg_model.predict(X_test_reg)

#evaluate
rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
r2 = r2_score(y_test_reg, y_pred_reg)

print("Linear Regression Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")
