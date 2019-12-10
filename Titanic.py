import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df


def process_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)
    df = create_dummies(df, "Age_categories")
    return df


def process_embarked(df):
    # train["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna("S")
    df = create_dummies(df, "Embarked")
    return df


def process_has_child(df):
    df["Has_child"] = (df["Parch"] > 0) & (df["Age"] >= 18)
    return df


def process_has_parent(df):
    df["Has_parent"] = (df["Parch"] > 0) & (df["Age"] < 18)
    return df


def process_is_alone(df):
    df["Alone"] = (df["Parch"] == 0) & (df["SibSp"] == 0)
    return df


def process_frame(df):
    cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
    label_names = ["Missing", 'Infant', "Child", 'Teenager', "Young Adult", 'Adult', 'Senior']
    df = process_age(df, cut_points, label_names)
    df = process_has_child(df)
    df = process_has_parent(df)
    df = process_is_alone(df)
    df = create_dummies(df, "Pclass")
    df = create_dummies(df, "Sex")
    df = create_dummies(df, "Embarked")
    return df


if __name__ == "__main__":

    # Load the training data and format it
    train_all = pd.read_csv("train.csv")
    train_all = process_frame(train_all)

    # List of features to use to train the model
    features = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
               'Age_categories_Missing', 'Age_categories_Infant',
               'Age_categories_Child', 'Age_categories_Teenager',
               'Age_categories_Young Adult', 'Age_categories_Adult',
               'Age_categories_Senior', 'Has_child', 'Has_parent', 'Alone',
               'Embarked_C', 'Embarked_Q', 'Embarked_S']

    # Split the data into training data and test data
    train_x, test_x, train_y, test_y = \
        train_test_split(train_all[features], train_all['Survived'], test_size=0.2, random_state=0)

    # Train the model
    lr = LogisticRegression()
    # scores = cross_val_score(lr, train_all[features], train_all['Survived'], cv=10)
    # score = np.mean(scores)

    lr.fit(train_x, train_y)

    # Check it's accuracy
    predictions = lr.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)

    stop_here = True
