import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(X_train[1])
plt.show()
# we are keeping the values in between 0-1 so that the weight of ANN will give better result
# as it does for similar values
X_train=X_train/255
X_test=X_test/255


print(X_train[5])


model=Sequential()
model.add(Flatten(input_shape=(28,28)))
# Dense layer
model.add(Dense(128,activation='relu'))#no need to mention no of inputs bcz the o/p of Flatten will be given to Dense
model.add(Dense(10,activation='softmax'))#this i sthe o/p layer we have 0-9 nodes as output
 
model.summary()
 
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam')
 
model.fit(X_train,y_train,epochs=10,validation_split=0.2)
 
y_prob=model.predict(X_test) #the output is the probability of happening 1/2/3/...
 
y_pred=y_prob.argmax(axis=1) #max value at which index
print(y_pred)
 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

 