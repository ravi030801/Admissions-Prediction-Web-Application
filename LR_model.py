# importing libraries
#independent: gre, gpa & rank dependent: admit

import numpy as np  #  mathmatical function 
import matplotlib.pyplot as plt  
import pandas as pd   # data analysis
import pickle
from sklearn.preprocessing import StandardScaler  # scaling
from sklearn.model_selection import train_test_split  # test/train 
from sklearn.linear_model import LogisticRegression   # model
 
 
#  importing dataset 

data_set=pd.read_csv("past_data.csv")
print(data_set.head())


# now we split the dataset into a training set and test set

x_train,x_test,y_train,y_test=train_test_split(data_set[['gre','gpa','rank']],data_set['admit'],test_size=0.5,random_state=0)


# now we do feature scaling because values are lie in diff ranges

st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.fit_transform(x_test)
# print(x_train[0:10,:])


# finally we are training our logistic regression model

model=LogisticRegression(random_state=0)
model.fit(x_train,y_train) # train the model on training set


# after training model its time to use it to do predictions on testing data.

y_pred=model.predict(x_test)


# confusion_matrix: use to check the model performance...

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix : \n",cm)

# getting accuracy
from sklearn.metrics import accuracy_score
print("Accuracy : ",accuracy_score(y_test,y_pred))


#  x_test=  gre gpa rank 
#            0   1   2 

# visualing the performance of our model.
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='coolwarm')
# plt.colorbar(label='Admit (1) or Not (0)')
plt.xlabel('GRE Scores')
plt.ylabel('GPA')

# Add a title to the plot
plt.title('Model Performance (Test Set)')

# Show the plot
plt.show()


# saving the model a pickle file
# pickle.dump(LogisticRegression,open('LR_model.pkl','wb'))
with open('LR_model.pkl','wb') as f:
    pickle.dump(model,f)

# loading the model to disk
# pickle.dump(LogisticRegression,'LR_model.pkl','rb')
with open('LR_model.pkl','rb') as f:
    load_model=pickle.load(f)
