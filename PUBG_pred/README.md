# PUBG
PlayerUnknown's BattleGrounds (PUBG) has enjoyed massive popularity. With over 50 million copies sold, it's the fifth best selling game of all time, and has millions of active monthly players. 
https://www.kaggle.com/c/pubg-finish-placement-prediction/overview
The team at PUBG has made official game data available for the public to explore and scavenge outside of "The Blue Circle." This competition is not an official or affiliated PUBG site - Kaggle collected data made possible through the PUBG Developer API.
We have been given over 65,000 games' worth of anonymized player data, split into training and testing sets, and asked to predict final placement from final in-game stats and initial player ratings.
In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.
We have created an ensemble model which predicts players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place).

## Ensemble Modelling
First we have done data processing, cleaning and wrangling like Dimesionality reduction using PCA, remove all the outliers. Then we split the dataset into Train and Validation dataset with the ratio of 0.5 and trained 1 LightGBM, 1 XGBoost, 1 CatBoost, 2 KNN, 1 SVC and 2 Neural Networks. After that we have created a new dataframe consisting of the Predictions from trained model on validation Set with the Ground truth Value. Then we Fit the Linear Regression Model to the New dataset. 

## File descriptions

- dataset/
  - train_V2.csv - Initial training set
  - test_V2.csv - Intial test set
  - train_V5.csv - Final training set
  - test_V5.csv - Final test set
  - final.csv - Ensemble predictions on Training set
  - submission_V2.csv - submission file used in the competetion

- model/
  - Download the pre trained models and extract the zip file using this google drive link
    https://drive.google.com/file/d/1CPzDpvaRhWRXoAFov6s4FZBXMR8Ybjat/view?usp=sharing
    Put them in the models folder
   
 - process/  contains all the data processing notebook 
   - answers.ipynb
   - inital_data_eng.ipynbv
   - data-processing-2.ipynb
   - final_processing.ipynb
   - submission.ipynb
   
 - variables/  contains all the variables used in the pickle format, so they can be reused.
   - num-max
   - win_kill
   - scaler
   - matchtype_enc
   
 ## How to make the predictions
   You can consider from the notebooks file present in the process folder. 
  
 ## How to Deploy the model
 Just install all the packages mentioned in the requirements.txt. Then open the cmd or terminal and run
 ````
 python main.py
 ````
