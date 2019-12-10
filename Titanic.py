import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------- Data Formatting & Feature extraction --------------------------------------------
def create_dummies(data_frame, feature_name):
    """Append a "one-hot" column to the data frame for each label within the specified column."""
    dummies = pd.get_dummies(data_frame[feature_name], prefix=feature_name)
    data_frame = pd.concat([data_frame, dummies], axis=1)
    return data_frame


def process_age(data_frame, bins, label_names):
    """Convert the age label into the specified age groups, each represented by a one-hot column."""
    data_frame["Age"] = data_frame["Age"].fillna(-0.5)
    data_frame["Age_categories"] = pd.cut(data_frame["Age"], bins, labels=label_names)
    data_frame = create_dummies(data_frame, "Age_categories")
    return data_frame


def process_embarked(data_frame):
    """Create a one-hot column for each embarking location."""
    # A very few entries are missing; we'll fill them with the most common departure location
    data_frame["Embarked"] = data_frame["Embarked"].fillna("S")
    data_frame = create_dummies(data_frame, "Embarked")
    return data_frame


def process_has_child(data_frame):
    """Check if the person is parent to someone on board."""
    data_frame["Has_child"] = (data_frame["Parch"] > 0) & (data_frame["Age"] >= 18)
    return data_frame


def process_has_parent(data_frame):
    """Check if the child is accompanied by a parent."""
    data_frame["Has_parent"] = (data_frame["Parch"] > 0) & (data_frame["Age"] < 18)
    return data_frame


def process_family_size(data_frame):
    """Check the number of family members onboard"""
    data_frame["Family_size"] = data_frame["Parch"] + data_frame["SibSp"]
    return data_frame


def process_deck(data_frame):
    """Process the deck number contained within the cabin"""
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "Z": 0}  # Z = Unknown
    data_frame["Cabin"] = data_frame["Cabin"].fillna('Z')
    decks = []
    for cabin_name in data_frame["Cabin"]:
        decks.append(cabin_name[0])
    data_frame["Deck"] = decks
    data_frame["Deck"] = data_frame["Deck"].map(deck)
    data_frame["Deck"] = data_frame["Deck"].fillna(0)  # Sometimes outright wrong deck numbers find their way in the data set
    data_frame = create_dummies(data_frame, "Deck")
    return data_frame


def process_fare_price(data_frame):
    """Process the fare price paid by the passenger. Discretize it in 4 quarters.
       Assume invalid data means person did not pay."""
    data_frame["Normalized_fare"] = pd.qcut(data_frame["Fare"], 4, labels=[0, 1, 2, 3])
    data_frame["Normalized_fare"] = data_frame["Normalized_fare"].fillna(0)
    return data_frame


def process_title(data_frame):
    """Isolate the title from the passenger's names."""
    # Extract titles
    data_frame['Title'] = data_frame['Name'].str.extract('([A-Za-z]+)\.', expand=False)

    # Replace titles with a more common title
    data_frame['Title'] = data_frame['Title'].replace('Mlle', 'Miss')
    data_frame['Title'] = data_frame['Title'].replace('Ms', 'Miss')
    data_frame['Title'] = data_frame['Title'].replace('Mme', 'Mrs')

    # Convert titles into numbers
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data_frame['Title_mapped'] = data_frame['Title'].map(titles)

    # Rare titles (Dr., Cpt., etc.) were turned into NaN; assign them a proper number.
    data_frame['Title_mapped'] = data_frame['Title_mapped'].fillna(5)

    return data_frame


def format_data(data_frame):
    """Formatting of the raw data for modeling : fill in missing info, create pertinent composite features and more."""
    # Split the passenger's ages into categories; bins obtained by manual observation of age's effect on survival rate
    cut_points = [-1, 0, 5, 12, 15, 60, 100]
    label_names = ["Missing", 'Infant', "Child", 'Teenager', 'Adult', 'Senior']
    data_frame = process_age(data_frame, cut_points, label_names)

    # Check the relations between the passengers
    data_frame = process_has_child(data_frame)
    data_frame = process_has_parent(data_frame)
    data_frame = process_family_size(data_frame)

    # Location of the passenger's cabin on the ship
    data_frame = process_deck(data_frame)
    data_frame = process_fare_price(data_frame)
    data_frame = process_title(data_frame)

    # Split discrete feature qualifiers into distinct columns
    data_frame = create_dummies(data_frame, "Sex")
    data_frame = create_dummies(data_frame, "Embarked")

    return data_frame


# List of features to use to train the model.
# Set as a global variable to be able to import it into other files
features = ['Pclass',
            'Sex_female', 'Sex_male',
            'Age_categories_Missing', 'Age_categories_Infant',
            'Age_categories_Adult',
            'Has_child', 'Has_parent', 'Family_size',
            'Embarked_C', 'Embarked_Q', 'Embarked_S',
            'Deck', "Normalized_fare", "Title_mapped"]


# ----------------------------------------------- Model Training -------------------------------------------------------
def fit_and_score(classifier, train_x, train_y, test_ratio=0.2, random_seed=0):
    """Use the provided features to fit the classifier. Function reserves a portion of data for testing purposes."""
    # Split the data into training and test sets
    train_x, test_x, train_y, test_y = \
        train_test_split(train_x, train_y, test_size=test_ratio, random_state=random_seed)

    # Train model
    classifier.fit(train_x, train_y)

    # Compute score
    pred_y = classifier.predict(test_x)
    accuracy = accuracy_score(test_y, pred_y)

    return classifier, accuracy


def cross_validation_score(classifier, train_x, train_y, num_of_slices=10):
    """Score a classifier using cross-validation (averaging the accuracy score obtained when fitting and testing
    different 'slices' of the training data)."""
    scores = cross_val_score(classifier, train_x, train_y, cv=num_of_slices)
    mean_score = np.mean(scores)
    return mean_score


def compare_classifier_types(train_x, train_y):
    """Compare the accuracy of a variety of classification methods."""
    # List the classifiers to test and their associated score
    classifiers = [LogisticRegression(),
                   RandomForestClassifier(n_estimators=400, random_state=0, criterion='gini', min_samples_leaf=1,
                                          min_samples_split=18),
                   DecisionTreeClassifier()]

    # Compute the accuracy score for each classifier
    scores = np.zeros(len(classifiers))
    for i, classifier in enumerate(classifiers):
        scores[i] = cross_validation_score(classifier, train_x, train_y)

    # Display the classifiers scores in descending order
    for i in np.argsort(-scores):
        print(classifiers[i].__class__.__name__ + f' : {scores[i]*100:.1f}%')


def tune_random_forest_parameter(train_x, train_y):
    """Run an exhaustive search for the best parameters for a random forest classifier in our application."""
    param_grid = {"criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5, 10, 25, 50, 70],
                  "min_samples_split": [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}

    rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=0, n_jobs=-1)
    clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
    clf.fit(train_x, train_y)
    print(clf.best_params_)


def generate_kagle_submission(classifier, features):
    """Load the test data, format it and then use the provided classifier to make predictions.
       Finally package the predictions in the format required by Kagle."""
    # Load and format the test data
    holdout = pd.read_csv("test.csv")
    holdout = format_data(holdout)

    # Create predictions using the provided classifier
    holdout_predictions = classifier.predict(holdout[features])

    # Create submission .csv file
    submission = pd.DataFrame({"PassengerId": holdout["PassengerId"],
                               "Survived": holdout_predictions})
    submission.to_csv('titanic_submission.csv', index=False)


# Chosen classifier
chosen_classifier = RandomForestClassifier(random_state=0, criterion='gini', n_estimators=400,
                                           min_samples_leaf=1, min_samples_split=18)


# ----------------------------------------------- Main operation -------------------------------------------------------
if __name__ == "__main__":
    # Load the training data and format it
    train_data = pd.read_csv("train.csv")
    train_data = format_data(train_data)

    # Compare a variety of classifier types
    # compare_classifier_types(train_data[features], train_data['Survived'])

    # According to the comparison, the 'Random Forest' gives the best results.
    # Tune it's hyper-parameters
    # tune_random_forest_parameter(train_data[features], train_data['Survived'])

    # Use it to generate the Kagle submission
    chosen_classifier = fit_and_score(chosen_classifier, train_data[features], train_data['Survived'])[0]
    generate_kagle_submission(chosen_classifier, features)
