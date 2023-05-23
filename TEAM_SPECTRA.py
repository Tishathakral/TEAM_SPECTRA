

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score

# Step 1: Data Cleaning
df = pd.read_csv('Leads-data.csv')
df.replace('9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0', float('nan'), inplace=True)
df_filtered = df[df['status'].isin(['WON', 'LOST'])].copy()  # Make a copy of the filtered DataFrame
categorical_columns = df_filtered.columns
imputer = SimpleImputer(strategy='most_frequent')
df_filtered.loc[:, categorical_columns] = imputer.fit_transform(df_filtered.loc[:, categorical_columns])


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score

# Encode categorical columns
categorical_cols = ['Agent_id', 'status', 'lost_reason', 'budget', 'duration', 'source', 'source_city', 'source_country',
                    'utm_source', 'utm_medium', 'des_city', 'des_country', 'room_type', 'lead_id']


# In[3]:


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_filtered[col] = df_filtered[col].astype(str)
    df_filtered[col] = le.fit_transform(df_filtered[col])
    label_encoders[col] = le


# In[4]:


# Split the data into training and testing sets
X = df_filtered.drop('budget', axis=1)
y = df_filtered['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


# In[6]:


# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert the predicted lead scores to integers
y_pred = y_pred.round().astype(int)

# Evaluate the model using F1-score
f1 = f1_score(y_test, y_pred, average='weighted')
print('F1-score:', f1)


# In[7]:


from sklearn.metrics import precision_score, recall_score, mean_absolute_error

# Calculate precision
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)

print('Precision:', precision)
print('Recall:', recall)
print('Mean Absolute Error:', mae)


# In[ ]:




