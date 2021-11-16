#### Create Loan Data for Classification in Python ####
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
import mlflow 
import argparse
from urllib.parse import urlparse

def parse_args():
    #max_depth=3, learning_rate=0.1, n_estimators=500
    parser = argparse.ArgumentParser(description="XGBoostClassifier example")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help="subsample ratio of columns when constructing each tree (default: 3.0)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=500,
        help="subs ample ratio of the training instances (default: 500)",
    )
    return parser.parse_args()

def main():
    # parse command-line arguments
    args = parse_args()    
    ColumnNames=['CIBIL','AGE', 'SALARY', 'APPROVE_LOAN']
    DataValues=[[480, 28, 610000, 1],
                 [480, 42, 140000, 0],
                 [480, 29, 420000, 0],
                 [490, 30, 420000, 0],
                 [500, 27, 420000, 0],
                 [510, 34, 190000, 0],
                 [550, 24, 330000, 1],
                 [560, 34, 160000, 1],
                 [560, 25, 300000, 1],
                 [570, 34, 450000, 1],
                 [590, 30, 140000, 1],
                 [600, 33, 600000, 1],
                 [600, 22, 400000, 1],
                 [600, 25, 490000, 1],
                 [610, 32, 120000, 1],
                 [630, 29, 360000, 1],
                 [630, 30, 480000, 1],
                 [660, 29, 460000, 1],
                 [700, 32, 470000, 1],
                 [740, 28, 400000, 1]]

    #Create the Data Frame
    LoanData=pd.DataFrame(data=DataValues,columns=ColumnNames)
    LoanData.head()

    #Separate Target Variable and Predictor Variables
    TargetVariable='APPROVE_LOAN'
    Predictors=['CIBIL','AGE', 'SALARY']
    X=LoanData[Predictors].values
    y=LoanData[TargetVariable].values

    #Split the data into training and testing set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # enable auto logging
    # mlflow.xgboost.autolog()

    ###################################################################
    ###### Xgboost Classification in Python #######

    with mlflow.start_run():
        clf=XGBClassifier(max_depth=args.max_depth, learning_rate=args.learning_rate, n_estimators=args.n_estimators, objective='binary:logistic', booster='gbtree')

        #Printing all the parameters of XGBoost
        print(clf)

        #Creating the model on Training Data
        clf.fit(X_train,y_train)
        prediction=clf.predict(X_test)

        #Measuring accuracy on Testing Data
        loss = log_loss(y_test, prediction)
        acc = accuracy_score(y_test, prediction)
        print( 'acc', acc)
        print('loss',loss)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(clf, "model", registered_model_name="xgboost")
        else:
            mlflow.sklearn.log_model(clf, "model", registered_model_name="xgboost")
main()
