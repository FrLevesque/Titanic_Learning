import numpy as np
import pandas as pd
from Titanic import process_deck, process_fare_price, process_title


def test_process_deck():
    # Setup a data frame with the expected values : correct decks, incorrect decks and null values
    data_frame = pd.DataFrame({"Cabin": ['A1', 'D', 'G10', 'T84', 'null']})
    data_frame.replace('null', np.NaN)

    # Test the function
    # Reminder : Deck numbers  = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "Z": 0}
    data_frame = process_deck(data_frame)
    assert((data_frame['Deck'] == np.array([1, 4, 7, 0, 0])).all())


def test_process_fare_price():
    # Setup a data frame with the expected values : numerical values and null values
    data_frame = pd.DataFrame({"Fare": [0, 1.2, 2, 3, np.NaN]})

    # Test the function
    data_frame = process_fare_price(data_frame)
    assert ((data_frame['Normalized_fare'] == np.array([0, 1, 2, 3, 0])).all())


def test_process_title():
    # Setup a data frame with the expected values : common and uncommon titles as well as null values
    data_frame = pd.DataFrame({"Name": ['Mr. Pink', 'Master. Lee', "Mlle. de Jonquiere", "Dr. Who", 'null']})
    data_frame.replace('null', np.NaN)

    # Test the function
    # Reminder : Titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data_frame = process_title(data_frame)
    assert((data_frame['Title_mapped'] == np.array([1, 4, 2, 5, 5])).all())
