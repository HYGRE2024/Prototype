# Prototype

Basically the Machine Learning model here trains the machine to predict crop yield over 30 Indian states namely:
'Assam' 'Karnataka' 'Kerala' 'Meghalaya' 'West Bengal' 'Puducherry' 'Goa'
 'Andhra Pradesh' 'Tamil Nadu' 'Odisha' 'Bihar' 'Gujarat' 'Madhya Pradesh'
 'Maharashtra' 'Mizoram' 'Punjab' 'Uttar Pradesh' 'Haryana'
 'Himachal Pradesh' 'Tripura' 'Nagaland' 'Chhattisgarh' 'Uttarakhand'
 'Jharkhand' 'Delhi' 'Manipur' 'Jammu and Kashmir' 'Telangana'
 'Arunachal Pradesh' 'Sikkim'
 
 Which takes attributes such as
 Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production',
 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield'
 
The Dataset used here is from kaggle the dataset is named Agricultural Crop Yield in Indian States Dataset(crop.csv).
One Hot Encoding is used here to categorize the values of specific columns.

The dataset is divided in training set and testing set for training the machine.

Models like LSTM,DecisionTree,Xgboost and Support Vector Machine were used to get the output.



1. Setting the Optimizer: 
   - `optimizer = Adam(learning_rate=0.001)`: This line creates an optimizer called Adam with a learning rate of 0.001. The optimizer is responsible for updating the model's weights during training to minimize the error.

2. Compiling the Model: 
   - `model.compile(optimizer=optimizer, loss='mean_absolute_error')`: This line compiles the model by specifying the optimizer to use (Adam) and the loss function ('mean_absolute_error'). The loss function measures how far off the model's predictions are from the actual values.

3. Model Summary: 
   - `model.summary()`: This line displays a summary of the model's architecture, showing the layers, the number of parameters, and other details.

4. Early Stopping Callback: 
   - `early_stopping = EarlyStopping(patience=5, restore_best_weights=True)`: This sets up a mechanism called early stopping. If the model doesn't improve after 5 consecutive epochs, training will stop early to prevent overfitting. It will also restore the best model weights observed during training.

5. Training the Model: 
   - `history = model.fit(...)`: This line starts the training process. The model learns from the training data (`X_train_ls` and `Y_train_ls`) over 150 epochs, with a batch size of 32. A portion of the training data (20%) is used for validation to monitor the model's performance on unseen data. The `callbacks=[early_stopping]` ensures that training stops early if necessary.

Overall, this code sets up and trains a machine learning model, with mechanisms to prevent overfitting and monitor training progress.

Then matplotlib was used to vizualize graph between actual and predicted 
