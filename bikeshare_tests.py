"""

Rohan Narasayya
CSE 163 Aj
This program tests the bikeshare_corr_tests and bikeshare_models
functions with a smaller version of the Capital Bikeshare
system dataset.
"""

import bikeshare
import bikeshare_models
import pandas as pd


def main(path):
    df = pd.read_csv(path)
    small_df = df[df['yr'] == 0]
    bikeshare.correlations(small_df)
    bikeshare.hypothesis_tests(small_df)
    bikeshare_models.predict_total_ridership(small_df)
    bikeshare_models.predict_casual_ridership(small_df)
    bikeshare_models.predict_registered_ridership(small_df)


if __name__ == '__main__':
    main(path = 'C:/Users/rohan/OneDrive/Documents/test/hour.csv')