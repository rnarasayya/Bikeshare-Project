"""

Rohan Narasayya
CSE 163 Aj
This program explores the Bikeshare data about the Capital Bikeshare
system in Washington D.C over a two-year period from 2011 to 2012 by
calculating and visualizing correlations between attribues and ridership.
It also looks for statistically significant differences in the data
betwen conditions of interest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import bikeshare_models

sns.set()


def correlations(df):
    """
    This function takes a dataframe and calculates the 
    correlation coefficient for each non-ridership column with each
    ridership type. Then it plots these attributes against the
    ridership type if the correlation is strong enough.
    """
    attr_type = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    cols = df.columns
    for i in range(len(attr_type)):
        cas_correlation = 0
        reg_correlation = 0
        cnt_correlation = 0
        if attr_type[i] == 0:
            labels, uniques = pd.factorize(df[cols[i]])
            ldf = pd.DataFrame(labels)
            cas_correlation = ldf[0].corr(df['casual'])
            reg_correlation = ldf[0].corr(df['registered'])
            cnt_correlation = ldf[0].corr(df['cnt'])
        else:
            cas_correlation = df[cols[i]].corr(df['casual'])
            reg_correlation = df[cols[i]].corr(df['registered'])
            cnt_correlation = df[cols[i]].corr(df['cnt'])
        if abs(cas_correlation) >= 0.30:
            print('The correlation between casual and ' + cols[i]
                  + ' is '+ str(cas_correlation))
            sns.relplot(data=df, x=cols[i], y='casual')
            plt.xlabel(cols[i])
            plt.ylabel('Casual Ridership')
            plt.title(cols[i] + ' vs Casual Ridership')
            path = 'C:/Users/rohan/OneDrive/Documents/test/' + cols[i] + \
                'cas_plot.png'
            plt.savefig(path, bbox_inches='tight')
        if abs(reg_correlation) >= 0.30:
            print('The correlation between registered and ' + cols[i]
                  + ' is ' + str(reg_correlation))
            sns.relplot(data=df, x=cols[i], y='registered')
            plt.xlabel(cols[i])
            plt.ylabel('Registered Ridership')
            plt.title(cols[i] + ' vs Registered Ridership')
            path = 'C:/Users/rohan/OneDrive/Documents/test/' + cols[i] + \
                'reg_plot.png'
            plt.savefig(path, bbox_inches='tight')
        if abs(cnt_correlation) >= 0.30:
            print('The correlation between cnt and ' + cols[i]
                  + ' is ' + str(cnt_correlation))
            sns.relplot(data=df, x=cols[i], y='cnt')
            plt.xlabel(cols[i])
            plt.ylabel('Total Ridership')
            plt.title(cols[i] + ' vs Total Ridership')
            path = 'C:/Users/rohan/OneDrive/Documents/test/' + cols[i] + \
                'cnt_plot.png'
            plt.savefig(path, bbox_inches='tight')


def hypothesis_tests(df):
    """
    This function takes a dataframe and performs 5 hypothesis tests
    for statistical significance between different conditions.
    """
    non_workday = df[df['workingday'] == 0]
    work_day = df[df['workingday'] == 1]
    non_workday_cas = non_workday.groupby('dteday')['casual'].sum()
    non_workday_cnt = non_workday.groupby('dteday')['cnt'].sum()
    non_workday_ratio = non_workday_cas / non_workday_cnt
    workday_cas = work_day.groupby('dteday')['casual'].sum()
    workday_cnt = work_day.groupby('dteday')['cnt'].sum()
    workday_ratio = workday_cas / workday_cnt
    t_value, p_value = stats.ttest_ind(non_workday_ratio, workday_ratio,
        alternative='greater')
    print('p-value is ' + str(p_value))
    alpha = .05
    if p_value < alpha:
        print('The non-working day ratio of casual riders is significantly' 
              ' greater than the working-day ratio of casual riders.')
    else:
        print('The non-working day ratio of casual riders is not'
              ' significantly greater than the working-day'
              ' ratio of casual riders.')
    inclement = df[(df['weathersit'] == 3) | (df['weathersit'] == 4)]
    non_inclement = df[(df['weathersit'] == 1) | (df['weathersit'] == 2)]
    inclement_cnt = inclement.groupby('dteday')['cnt'].sum()
    non_inclement_cnt = non_inclement.groupby('dteday')['cnt'].sum()
    t_value, p_value = stats.ttest_ind(inclement_cnt, non_inclement_cnt,
        alternative='less')
    print('p-value is ' + str(p_value))
    if p_value < alpha:
        print('Total ridership does drop significantly in inclement'
              ' weather.')
    else:
        print('Total ridership does not drop significantly in'
              ' inclement weather.')
    winter = df[df['season'] == 1]
    spring = df[df['season'] == 2]
    summer = df[df['season'] == 3]
    fall = df[df['season'] == 4]
    winter_cnt = winter['cnt']
    spring_cnt = spring['cnt']
    summer_cnt = summer['cnt']
    fall_cnt = fall['cnt']
    t_value, p_value = stats.ttest_ind(summer_cnt, winter_cnt,
        alternative='greater')
    print('p-value is ' + str(p_value))
    if p_value < alpha:
        print('There is significantly more total ridership in'
              ' the summer than in the winter.')
    else:
        print('There is not significantly more total ridership'
              ' in the summer than in the winter.')
    t_value, p_value = stats.ttest_ind(summer_cnt, spring_cnt,
        alternative='greater')
    print('p-value is ' + str(p_value))
    if p_value < alpha:
        print('There is significantly more total ridership in'
              ' the summer than in the spring.')
    else:
        print('There is not significantly more total ridership'
              ' in the summer than in the spring.')
    t_value, p_value = stats.ttest_ind(summer_cnt, fall_cnt,
        alternative='greater')
    print('p-value is ' + str(p_value))
    if p_value < alpha:
        print('There is significantly more total ridership in'
              ' the summer than in the fall.')
    else:
        print('There is not significantly more total ridership'
              ' in the summer than in the fall.')


def main(path):
    df = pd.read_csv(path)
    correlations(df)
    hypothesis_tests(df)
    bikeshare_models.predict_casual_ridership(df)
    bikeshare_models.predict_registered_ridership(df)
    bikeshare_models.predict_total_ridership(df)


if __name__ == '__main__':
    main(path = 'C:/Users/rohan/OneDrive/Documents/test/hour.csv')