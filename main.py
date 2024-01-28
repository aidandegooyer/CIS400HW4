import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Remove id and number
train_data = train_data.iloc[:, 2:]
test_data = test_data.iloc[:, 2:]

# Input missing values with the mean
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Split features and labels
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]

basic_model = Sequential()
basic_model.add(Dense(units=input_dim, activation='sigmoid', input_dim=input_dim))
basic_model.add(Dense(units=input_dim * 2, activation='sigmoid'))
basic_model.add(Dense(units=1, activation='sigmoid'))

optimizer = Adam(clipvalue=0.5)
basic_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


history = basic_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

loss, accuracy = basic_model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
