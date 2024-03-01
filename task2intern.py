#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# path of the dataset
titanic_df = pd.read_csv(r"C:\Users\TRINITY ELE\Downloads\titanic\train.csv")


print(titanic_df.head())


# In[6]:


print(titanic_df.isnull().sum())
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

titanic_df.drop('Cabin', axis=1, inplace=True)

print(titanic_df.isnull().sum())


# In[7]:


sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title('Survival Count by Sex')
plt.show()

# class
sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
plt.title('Survival Count by Ticket Class')
plt.show()

# age group
titanic_df['AgeGroup'] = pd.cut(titanic_df['Age'], bins=[0, 12, 18, 30, 50, 100], labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Elderly'])
sns.countplot(x='AgeGroup', hue='Survived', data=titanic_df)
plt.title('Survival Count by Age Group')
plt.show()

# number of siblings/spouses
sns.countplot(x='SibSp', hue='Survived', data=titanic_df)
plt.title('Survival Count by Siblings/Spouses')
plt.show()

# number of parents/children
sns.countplot(x='Parch', hue='Survived', data=titanic_df)
plt.title('Survival Count by Parents/Children')
plt.show()

#passengers who survived and who didn't
sns.boxplot(x='Survived', y='Fare', data=titanic_df)
plt.title('Fare Distribution of Survived vs. Not Survived')
plt.show()

sns.countplot(x='Embarked', hue='Survived', data=titanic_df)
plt.title('Survival Count by Port of Embarkation')
plt.show()


# In[ ]:




