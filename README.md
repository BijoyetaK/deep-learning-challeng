# Module 21 Challenge - Alphabet Soup Charity - Deep Learning
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. This project uses the features in the provided dataset and creates a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

# Dataset: 
   * Within this dataset are a number of columns that capture metadata about each organization, such as:
   * EIN and NAME—Identification columns
   * APPLICATION_TYPE—Alphabet Soup application type
   * AFFILIATION—Affiliated sector of industry
   * CLASSIFICATION—Government organization classification
   * USE_CASE—Use case for funding
   * ORGANIZATION—Organization type
   * STATUS—Active status
   * INCOME_AMT—Income classification
   * SPECIAL_CONSIDERATIONS—Special considerations for application
   * ASK_AMT—Funding amount requested
   * IS_SUCCESSFUL—Was the money used effectively(1 or 0)
     

# Data Pre processing
  ### Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

  * Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
    * What variable(s) are the target(s) for your model?
      * IS_SUCCESSFUL is used as the target. 
      * What variable(s) are the feature(s) for your model?
      * APPLICATION_TYPE,AFFILIATION,CLASSIFICATION,USE_CASE,ORGANIZATION,STATUS,INCOME_AMT,SPECIAL_CONSIDERATIONS,ASK_AMT columns were used as inout features.
      * EIN and NAME columns were dropped because they are have no role in the result. 
      * Determine the number of unique values for each column using value_counts.
      * Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
      * Binning is applied on 2 columns, APPLICATION_TYPE and CLASFFICATION.
        Value of 156 is used as a cut off for APPLICATION_TYPE, while value 1883 was used as cut off for CLASFFICATION into 'Other'. 
      * Use pd.get_dummies() to encode categorical variables.
      * Split the preprocessed data into a features array, X, and a target array, y.Use these arrays and the train_test_split function to split the data into training and testing datasets.
      * Used StandardScaler to scale the training and testing features by doing fit_transform:  X_train_scaled and X_test_scaled. 

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

# Compile, Train, and Evaluate the Model
* How many neurons, number of hidden layers, number of nodes for each layer and activation function selected for the neural network?
  #### Compile
  * Approach 1: Tried using Keras tuner approach to just check the optimum hyperparameter options for training the model: (not training yet)
    
    * The tuner search gave a accuracy close 0.7245 but the best value selected was:
      ![image](https://github.com/BijoyetaK/deep-learning-challeng/assets/126313924/12cd70d5-45f1-459c-b77b-2190293f8f0a)
    * Best model hyperparameters selected by Keras tuner:
      Number of hidden layers: 2, first layer - 9 neurons, Each hidden layer - 5 neurons, optimum epochs 7.
    * Evaluating the best model selected by keras tuner against the full test data: X_test_scaled,y_test gave the below accuracy and loss:
      ![image](https://github.com/BijoyetaK/deep-learning-challeng/assets/126313924/e90fe18e-d689-4703-a263-d5ec29abcaef)
      
  * Approach 2: Just going by intuition and selecting 
    * As per thumb rule - since the number of input features are 43, hence the number of neurons are varied from 1 to 120 for first hidden layer 1.
    * 85 neurons for the second hidden later was simply at random.
    * First layer - Activation function -> 'LeakyReLU'
    * Second layer - Activation function -> 'ReLU'
    * Output layer - Activation function -> 'tanh'
    * Compiled the model using loss as "binary_crossentropy", since this is a binary classification, optimizer used as "adam" and accuracy metrics are captured.
  #### Train   
    * Trained the model using X_train_scaled,y_train and epochs of 100.
  #### Evaluate
    * Evaluating the model using test data(X_test_scaled,y_test):
      Loss: 0.5668252110481262, Accuracy: 0.7244315147399902
         
# Training History 
  * Stored the model training history into a dataframe for plotting loss and accuracy.
    ![image](https://github.com/BijoyetaK/deep-learning-challeng/assets/126313924/0a6ca654-7fb9-4334-8266-430a6eb666fc)

    ![image](https://github.com/BijoyetaK/deep-learning-challeng/assets/126313924/4b05db5b-c39f-464b-85d5-59d3bc282ab9)

    ![image](https://github.com/BijoyetaK/deep-learning-challeng/assets/126313924/ed4edfc7-48ef-422d-95db-4601a3cd1203)

# Optimize the model
  * Were you able to achieve the target model performance?
    - Maximum attained accuracy using randomly selected neurons was 72.4% which is almost equal to the accuracy of the best optimized mode evaluated by Keras tuner. 


#  Summary
Based on the nature of the graphs observed , I think that the best accuracy model has been attained indicating that changing the number of epochs may not play a crucial role, because Keras selected 7 epochs as optimum yet accuracy remains the same even after 100 epochs.The curve (loss and accuracy) may be improved by changing the number of neurons, changing the number of hidden layers and changing the activation functions. However, overall pattern indicates it may be very difficult to increase the accuracy further. 

#  Recommendation
Since there are quite a higher number of input parameters, a random forest classifier could give better accuracy in the context of deep learning. Will have to experiment a bit. :) 

#  References
   * https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3#:~:text=The%20number%20of%20hidden%20neurons%20should%20be%202%2F3%20the,size%20of%20the%20input%20layer.
   * https://www.researchgate.net/post/How-to-decide-the-number-of-hidden-layers-and-nodes-in-a-hidden-layer
   * https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
   * https://machinelearningmastery.com/evaluate-skill-deep-learning-models/
   * https://www.linkedin.com/pulse/choosing-number-hidden-layers-neurons-neural-networks-sachdev/
     
