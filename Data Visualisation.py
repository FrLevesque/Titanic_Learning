import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def binned_survival_ratio(train_data, feature_name, bins):
    """Compute the survival ratio according to the specified feature. Bin the results as requested."""
    # Compute in which bin each data point falls
    indices = np.digitize(train_data[feature_name], bins, right=True) - 1

    # Compute the total number of cases falling within each bin as well as the survivor number
    total_count = np.zeros(len(bins) - 1)
    survivor_count = np.zeros(len(bins) - 1)
    for df_index, bin_index in enumerate(indices):
        total_count[bin_index] += 1
        survivor_count[bin_index] += train_data.iloc[df_index]["Survived"]

    # Return the binned survival rate
    return survivor_count / total_count


def age_effect(train_data):
    """Display the effect of age on the survival rate"""
    # Only keep valid data points
    valid_passenger = train_data[train_data["Age"].notnull()]
    valid_males = valid_passenger[valid_passenger["Sex"] == 'male']
    valid_females = valid_passenger[valid_passenger["Sex"] == 'female']

    # Separate the passengers into age bins
    bin_width = 2.0  # years
    upper_age_limit = 80.0
    bins = np.linspace(0.0, upper_age_limit, math.floor(upper_age_limit/bin_width) + 1)
    bin_centers = (bins[:-1] + bins[1:])/2

    # Compute each age bin survival rate
    total_survival_rate = binned_survival_ratio(valid_passenger, 'Age', bins)
    male_survival_rate = binned_survival_ratio(valid_males, 'Age', bins)
    female_survival_rate = binned_survival_ratio(valid_females, 'Age', bins)

    # Plot it out
    fig, axs = plt.subplots(3)
    axs[0].bar(bin_centers, total_survival_rate*100)
    axs[0].set_title('All passengers')
    axs[0].set_xlim([0, upper_age_limit])

    axs[1].bar(bin_centers, male_survival_rate*100)
    axs[1].set_title('Males')
    axs[1].set_ylabel('Survival rate (%)')
    axs[1].set_xlim([0, upper_age_limit])

    axs[2].bar(bin_centers, female_survival_rate*100)
    axs[2].set_title('Females')
    axs[2].set_xlabel('Age (years)')
    axs[2].set_xlim([0, upper_age_limit])
    plt.show()


def sex_effect(train_data):
    """Display the effect of gender on the survival rate"""
    # Compute the various survival rates
    males = train_data[train_data["Sex"] == 'male']
    males_survival_rate = males['Survived'].sum() / males.shape[0]

    females = train_data[train_data["Sex"] == 'female']
    females_survival_rate = females['Survived'].sum() / females.shape[0]

    total_survival_rate = train_data['Survived'].sum() / train_data.shape[0]

    # Plot it out
    plt.bar(['Male', 'Female', 'Total'], [males_survival_rate*100, females_survival_rate*100, total_survival_rate*100])
    plt.title('Survival rate vs. Passenger\'s sex')
    plt.ylabel('Survival rate (%)')
    plt.show()


def embarking_location_effect(train_data):
    """Display the effect of the embarking location on the survival rate"""
    # Only keep valid data points
    train_data = train_data[train_data["Embarked"].notnull()]

    # Compute the various survival rates
    c = train_data[train_data["Embarked"] == 'C']
    c_survival_rate = c['Survived'].sum() / c.shape[0]

    q = train_data[train_data["Embarked"] == 'Q']
    q_survival_rate = q['Survived'].sum() / q.shape[0]

    s = train_data[train_data["Embarked"] == 'S']
    s_survival_rate = s['Survived'].sum() / s.shape[0]

    total_survival_rate = train_data['Survived'].sum() / train_data.shape[0]

    # Plot it out
    plt.bar(['C', 'Q', 'S', 'Total'],
            [c_survival_rate * 100, q_survival_rate * 100, s_survival_rate * 100, total_survival_rate * 100])
    plt.title('Survival rate vs. Embarked location')
    plt.ylabel('Survival rate (%)')
    plt.show()


def cabin_effect(train_data):
    """Display the effect of the assigned cabin deck on the survival rate"""
    # Process the data
    from Titanic import process_deck
    train_data = process_deck(train_data)

    # Compute the survival rate for each deck
    decks = ['Unknown', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Total']
    decks_survival_rate = np.zeros(len(decks))
    for i in range(len(decks)-1):
        deck_data = train_data[train_data["Deck"] == i]
        decks_survival_rate[i] = deck_data['Survived'].sum() / deck_data.shape[0]

    # Total survival rate
    decks_survival_rate[-1] = train_data['Survived'].sum() / train_data.shape[0]

    # Plot it out
    plt.bar(decks, decks_survival_rate * 100)
    plt.title('Survival rate vs. Cabin deck')
    plt.ylabel('Survival rate (%)')
    plt.show()


def passenger_class_effect(train_data):
    """Display the effect of the passenger's class on the survival rate"""
    # Only keep valid data points
    train_data = train_data[train_data["Pclass"].notnull()]

    # Compute the survival rate for passenger class
    passenger_class = ['1', '2', '3', 'Total']
    class_survival_rate = np.zeros(len(passenger_class))
    for i in range(len(passenger_class)-1):
        passenger_class_data = train_data[train_data["Pclass"] == (i + 1)]
        class_survival_rate[i] = passenger_class_data['Survived'].sum() / passenger_class_data.shape[0]

    # Total survival rate
    class_survival_rate[-1] = train_data['Survived'].sum() / train_data.shape[0]

    # Plot it out
    plt.bar(passenger_class, class_survival_rate * 100)
    plt.title('Survival rate vs. Passenger class')
    plt.ylabel('Survival rate (%)')
    plt.show()


def fare_price_effect(train_data):
    """Display the effect of the passenger's ticket price on the survival rate"""
    # Only keep valid data points
    train_data = train_data[train_data["Fare"].notnull()]

    # Separate the passengers fare price into bins
    bin_width = 25.0
    bins = np.linspace(0.0, 520.0, math.floor(520.0/bin_width) + 1)
    bin_centers = (bins[:-1] + bins[1:])/2

    # Compute each age bin survival rate
    survival_rates = binned_survival_ratio(train_data, 'Fare', bins)

    # Plot the result
    plt.bar(bin_centers, survival_rates * 100)
    plt.title('Survival rate vs. Fare price')
    plt.xlabel('Fare price ($)')
    plt.ylabel('Survival rate (%)')
    plt.show()


def parenthood_effect(train_data):
    """Display the effect of having children (for adults) or parents (for children) on the survival rate"""
    # Only keep valid data points
    train_data = train_data[train_data["Age"].notnull() & train_data["Parch"].notnull()]

    # Check a few family relationships
    from Titanic import process_has_child, process_has_parent
    train_data = process_has_child(train_data)
    train_data = process_has_parent(train_data)

    # Compare survival rate for adults with / without child
    adults = train_data[train_data["Age"] >= 18]
    adults_with_child = adults[adults["Has_child"] == 1]
    adults_without_child = adults[adults["Has_child"] == 0]
    awc_survival_rate = adults_with_child['Survived'].sum() / adults_with_child.shape[0]
    awoc_survival_rate = adults_without_child['Survived'].sum() / adults_without_child.shape[0]

    # Do the same for the children
    children = train_data[train_data["Age"] < 12]
    children_with_parent = children[children["Has_parent"] == 1]
    children_without_parent = children[children["Has_parent"] == 0]
    cwp_survival_rate = children_with_parent['Survived'].sum() / children_with_parent.shape[0]
    cwop_survival_rate = children_without_parent['Survived'].sum() / children_without_parent.shape[0]

    # Plot it
    plt.figure()
    plt.bar(['Parent', 'Non-parent', 'Child w. parent', 'Child w/o. parent'],
            [awc_survival_rate * 100, awoc_survival_rate * 100, cwp_survival_rate * 100, cwop_survival_rate * 100])
    plt.title('Survival rate vs. Parenthood')
    plt.ylabel('Survival rate (%)')
    plt.show()


def family_size_effect(train_data):
    """Display the effect of the total family size onboard on the survival rate"""
    # Only keep valid data points
    train_data = train_data[train_data["SibSp"].notnull() & train_data["Parch"].notnull()]

    # Compare survival rates according to the total number of family members on board
    train_data["Family_size"] = train_data["Parch"] + train_data["SibSp"]
    family_size = np.arange(0, 7)
    family_survival_rate = np.zeros(len(family_size))
    for i in range(len(family_size)):
        family_data = train_data[train_data["Family_size"] == i]
        family_survival_rate[i] = family_data['Survived'].sum() / family_data.shape[0]

    # Plot it out
    plt.figure()
    plt.bar(family_size, family_survival_rate * 100)
    plt.title('Survival rate vs. Family size')
    plt.ylabel('Number of family members onboard')
    plt.ylabel('Survival rate (%)')
    plt.show()


def random_forest_feature_importance(train_data):
    from Titanic import fit_and_score, format_data, features, chosen_classifier

    train_data = format_data(train_data)
    chosen_classifier = fit_and_score(chosen_classifier, train_data[features], train_data['Survived'])[0]

    importances = pd.DataFrame(
       {'feature': train_data[features].columns, 'importance': np.round(chosen_classifier.feature_importances_*100, 1)})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    importances.plot.bar()
    plt.ylabel("Importance (%)")
    plt.xlabel("Feature")
    plt.show()


if __name__ == "__main__":
    # Load the data
    data_frame = pd.read_csv("train.csv")

    # Display the effects of various data features
    age_effect(data_frame)
    sex_effect(data_frame)
    embarking_location_effect(data_frame)
    cabin_effect(data_frame)
    passenger_class_effect(data_frame)
    fare_price_effect(data_frame)
    parenthood_effect(data_frame)
    family_size_effect(data_frame)

    # Illustrate the main features used by the chosen classifier
    random_forest_feature_importance(data_frame)
