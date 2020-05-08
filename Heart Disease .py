#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv(r"C:\Users\LENOVO\Desktop\Heart.csv")


#  Lets see first five rows of our dataset.
# 

# In[6]:


df.head()


# Let see how many patients are suffering from heart disease in dataset.

# In[7]:


df.target.value_counts()


# In[12]:


noofpatientshavingdisease=len(df[df.target==1])
noofpatientsnothavingdisease=len(df[df.target==0])
totalpatients=len(df.target)
print("percentage of patients suffering from heart disease: {:.4f}%".format((noofpatientshavingdisease/totalpatients)*100))
print("percentage of patients not suffering from heart disease: {:.4f}%".format((noofpatientsnothavingdisease/totalpatients)*100))


# let visualize with respect to gender

# In[14]:


sns.countplot(x="sex",data=df)
plt.xlabel("0=female,1=male")
plt.show()


# In[33]:


pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,5))
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.legend(["Not having Disease", "Having Disease"])
plt.ylabel('Frequency with respect to Sex')
plt.xticks(rotation=0)
plt.show()


# lets see with respect to age.(frequency table)

# In[53]:


pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,5))
plt.title("Heart Disease frequency for ages")
plt.xlabel("Ages")
plt.legend(["Not having Disease", "Having Disease"])
plt.ylabel("Frequency with respect to ages")
plt.show()


# In[30]:


#thalach : (here maximum heart rate)
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# lets see with respect to blood sugar

# In[43]:


pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6))
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS > 120 mg/dl (0 = False, 1 = True)')
plt.legend(["Not having Disease", "Having Disease"])
plt.ylabel('Frequency with respect to FBS')
plt.xticks(rotation=0)
plt.show()


# In[52]:


#trestbps(blood pressure)
pd.crosstab(df.trestbps,df.target).plot(kind="bar",figsize=(20,8))
plt.title('Heart disease according to Blood Pressure')
plt.xlabel('Blood pressure')
plt.xticks(rotation=0)
plt.legend(["Not having Disease", "Having Disease"])
plt.ylabel('Frequency with respect to Blood Pressure')
plt.show()


# In[48]:



pd.crosstab(df.restecg,df.target).plot(kind="bar",figsize=(15,6))
plt.title('Heart Disease Frequency According To restecg')
plt.xlabel('restecg - resting electrocardiographic results ,(0-2)' )
plt.xticks(rotation = 0)
plt.legend(["Not having Disease", "Having Disease"])
plt.ylabel('Frequency with repect to restecg')
plt.show()


# In[51]:


pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6))
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type ,(0-3)')
plt.xticks(rotation = 0)
plt.ylabel('Frequency with respect to cp')
plt.show()


# In[ ]:




