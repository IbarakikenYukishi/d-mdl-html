import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functions import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import os
import csv
import requests
import math

# data source
#data_url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'


filename = 'covid19.csv'

#open(filename, 'wb').write(requests.get(data_url).content)

df_all = pd.read_csv(filename)

category = 'cases'  # we only studied the time series of cases

# list of countries with no less than 10,000 cumulative cases by Apr. 30
candidates_country = ['Austria', 'Belarus', 'Belgium', 'Brazil',
                      'Canada', 'Chile', 'China', 'Ecuador', 'France', 'Germany',
                      'India', 'Indonesia', 'Iran', 'Ireland', 'Israel', 'Italy', 'Japan',
                      'Mexico', 'Netherlands',
                      'Pakistan', 'Peru', 'Poland', 'Portugal',
                      'Qatar', 'Romania', 'Russia', 'Saudi_Arabia',
                      'Singapore', 'South_Korea', 'Spain', 'Sweden',
                      'Switzerland', 'Turkey', 'Ukraine', 'United_Arab_Emirates',
                      'United_Kingdom', 'United_States_of_America', 'Tokyo']

df_candidates = pd.DataFrame(columns=['country', 'date', 'cases'])
for index, row in df_all.iterrows():
    if row['countriesAndTerritories'] in candidates_country:
        date = datetime(row['year'], row['month'], row['day'])
        df_candidates = df_candidates.append({'country': row['countriesAndTerritories'], 'date': date,
                                              'cases': row['cases']},
                                             ignore_index=True)

# merge data
df_candidates = df_candidates.reset_index(drop=True)

# We calcuated the delta using the inital event in Japan and applied it to all the other countries
# The initial warning for Japan was set on Feb. 27 2020 when the government required closing elementary, junior high, and high schools
# We were concerned several days before the date of the initial event
# since there may exist delays in announcing the event

t = np.arange(datetime(2020, 2, 17), datetime(2020, 2, 28), timedelta(days=1))

# daily new cases
case_values = []
for date_tmp in t:
    row = df_candidates.loc[(df_candidates['country'] == 'Japan') & (
        df_candidates['date'] == date_tmp)]
    if len(row[category].values) > 0 and (row[category].values[0] > 0):
        case_values.append(row[category].values[0])
    else:
        case_values.append(0)

# Calculating the delta for the 1st D-MDL
for i in range(5, len(case_values)):
    score, _ = MDL_gaussian_1th(case_values[i - 5:i])
    delta = delta_First_NML(score, 2, 5)
    #print('change score:{0:.4f}, delta:{1:.4f}'.format(score, delta))

# Calculating the delta for the 2nd D-MDL
for i in range(6, len(case_values)):
    score, _ = MDL_gaussian_2nd(case_values[i - 6:i])
    delta = delta_Second_NML(score, 2, 6)
    #print('change score:{0:.4f}, delta:{1:.4f}'.format(score, delta))

# All the change scores that are less than zero are not meaningful
# All the delta are calcuated to be larger than one
# In the following experiment, it was set to be 0.99 because of the
# concept of confidence parameter

# The date on which the social distancing was implemented for individual
# countries
event_dict = {}
event_dict['Austria'] = datetime(2020, 3, 16)
event_dict['Belarus'] = datetime(2020, 4, 9)
event_dict['Belgium'] = datetime(2020, 3, 18)
event_dict['Brazil'] = datetime(2020, 3, 24)
event_dict['Canada'] = datetime(2020, 3, 17)
event_dict['Chile'] = datetime(2020, 3, 26)
event_dict['China'] = datetime(2020, 1, 23)
#event_dict['Ecuador'] = datetime(2020, 3, 16)
event_dict['France'] = datetime(2020, 3, 17)
event_dict['Germany'] = datetime(2020, 3, 16)
event_dict['India'] = datetime(2020, 3, 25)
event_dict['Indonesia'] = datetime(2020, 4, 6)
event_dict['Iran'] = datetime(2020, 3, 24)
event_dict['Ireland'] = datetime(2020, 3, 12)
event_dict['Israel'] = datetime(2020, 3, 15)
event_dict['Italy'] = datetime(2020, 3, 9)
event_dict['Japan'] = datetime(2020, 4, 7)
event_dict['Mexico'] = datetime(2020, 3, 23)
event_dict['Netherlands'] = datetime(2020, 3, 15)
event_dict['Pakistan'] = datetime(2020, 3, 24)
event_dict['Peru'] = datetime(2020, 3, 16)
event_dict['Poland'] = datetime(2020, 3, 24)
event_dict['Portugal'] = datetime(2020, 3, 19)
event_dict['Qatar'] = datetime(2020, 3, 23)
event_dict['Romania'] = datetime(2020, 3, 23)
event_dict['Russia'] = datetime(2020, 3, 30)
event_dict['Saudi_Arabia'] = datetime(2020, 3, 24)
event_dict['Singapore'] = datetime(2020, 4, 7)
event_dict['South_Korea'] = datetime(2020, 2, 25)
event_dict['Spain'] = datetime(2020, 3, 13)
event_dict['Sweden'] = datetime(2020, 3, 24)
event_dict['Switzerland'] = datetime(2020, 3, 16)
event_dict['Turkey'] = datetime(2020, 3, 21)
event_dict['Ukraine'] = datetime(2020, 3, 25)
event_dict['United_Arab_Emirates'] = datetime(2020, 3, 31)
event_dict['United_Kingdom'] = datetime(2020, 3, 24)
event_dict['United_States_of_America'] = datetime(2020, 3, 19)

# City and province level
# This could be removed if you want to remove Tokyo from the
# calculation of statistics.
event_dict['Tokyo'] = datetime(2020, 4, 7)

set_no_event = []
for country in candidates_country:
    if country not in event_dict:
        set_no_event.append(country)

delta = 0.05  # delta for 0th
delta_first = 0.99  # delta for 1st
delta_second = 0.99  # delta for 2nd
dimension = 2  # dimension of the univariate Gaussian
font_size = 36  # figure font size
output_path = './data/gaussian_figs/'  # output path of the figures
interval = 31
os.makedirs(output_path, exist_ok=True)

alarms_for_each_country = pd.DataFrame()

# t = np.arange(datetime(2020, 1, 1), datetime(2020, 5, 1),
#              timedelta(days=1))  # The timeline for online detection
num_change_dict = {}  # number of change points before event
num_0th_change_dict = {}  # number of 0th change points before event
num_1st_change_dict = {}  # number of 1st change points before event
num_2nd_change_dict = {}  # number of 2nd change points before event

# the day on which the first decrease change point raised by 0th for
# countries with event
decrease_dict = {}
# the day on which the first decrease change raised by 0th for countries
# without event
decrease_no_event_dict = {}

# days between the 0th change and the first either 1th or 2nd change
# within the window
days_between_zero_earliest = []
# days between the 0th change and the first 1th change within the window
days_between_zero_first = []
# days between the 0th change and the first 2nd change within the window
days_between_zero_second = []

stat_interval = 31

first_candidate_num = 0  # candicate outbreak allowing for 1st sign detection
first_detected_num = 0  # number of outbreaks whose signs are detected by 1st D-MDL
second_candidate_num = 0  # candicate outbreak allowing for 2nd sign detection
second_detected_num = 0  # number of outbreaks whose signs are detected by 2nd D-MDL
# number of outbreaks whose signs are detected by either 1st D-MDL or 2nd D-MDL
detected_num = 0
candidate_num = 0  # candicate outbreak allowing for both 1st and 2nd sign detection


for country in candidates_country:
    print('country:', country)
    if country in event_dict:
        dates_events = event_dict[country]

    df_date = df_candidates[df_candidates['country'] == country]['date']
    max_df_date = max(df_date)

    stat_date_end = np.datetime64(max_df_date)
    stat_date_start = np.datetime64(
        max_df_date - timedelta(days=stat_interval))
    min_df_date = np.datetime64(min(df_date))

    t = np.arange(datetime(2020, 1, 1), max_df_date + timedelta(days=1),
                  timedelta(days=1))  # The timeline for online detection

    # Determine the starting point of the detection
    # Dates on which there are no daily new cases and the avearage daily new
    # cases over the past 10 ten days is less than one is skipped.
    t_candidates = []
    t_num = []
    for date_tmp in t:
        row = df_candidates.loc[(df_candidates['country'] == country) & (
            df_candidates['date'] == date_tmp)]
        if len(row[category].values) > 0 and (row[category].values[0] > 0):
            t_candidates.append(date_tmp)
            t_num.append(row[category].values[0])
        else:
            if len(t_num) <= 10 and np.mean(t_num) <= 1:
                t_candidates = []
                t_num = []
            else:
                t_candidates.append(date_tmp)
                t_num.append(0)

    case_values = {}
    case_max = 0
    max_date = 0
    for date_tmp in t:
        if date_tmp in t_candidates:
            row = df_candidates.loc[(df_candidates['country'] == country) & (
                df_candidates['date'] == date_tmp)]
            if len(row[category].values) > 0 and (row[category].values[0] > 0):
                case_values[date_tmp] = row[category].values[0]
                if case_max < row[category].values[0]:
                    case_max = row[category].values[0]
                    max_date = date_tmp
            else:
                case_values[date_tmp] = 0

    dict_changes_cases = {}
    dic_windows_cases = {}
    MDL_scores_cases = []
    dates_tmp_cases = []
    windows_cases = []

    dict_first_changes_cases = {}
    dict_first_windows_cases = {}
    MDL_first_scores_cases = []
    dates_tmp_cases_first = []

    dict_second_changes_cases = {}
    dict_second_windows_cases = {}
    MDL_second_scores_cases = []
    dates_tmp_cases_second = []

    window_minimal = 4
    window_size = 0
    num_changes_before_event = 0
    num_0th_changes_before_event = 0
    num_1st_changes_before_event = 0
    num_2nd_changes_before_event = 0
    for date_tmp in t:
        if date_tmp in t_candidates:
            dates_tmp_cases.append(date_tmp)
            dates_tmp_cases_first.append(date_tmp)
            dates_tmp_cases_second.append(date_tmp)

            # 0th D-MDL
            if window_size >= window_minimal:
                t_tmp = np.arange(date_tmp - np.timedelta64(window_size - 1, 'D'), date_tmp +
                                  np.timedelta64(1, 'D'), timedelta(days=1))  # .astype(datetime)
                tmp_case_values = []
                for d in t_tmp:
                    if d in case_values:
                        tmp_case_values.append(case_values[d])
                    else:
                        tmp_case_values.append(0)

                MDL_max, cut_point = MDL_gaussian_0th(tmp_case_values)
                MDL_scores_cases.append(MDL_max)

                windows_cases.append(window_size)
                threshold_tmp = threshod_MDL(delta, dimension, window_size)
                # print('date:{0}, window size:{1}, threshold:{2}, MDL score:{3}'.format(
                #    date_tmp, window_size, threshold_tmp, MDL_max))

                #### first order ####
                if window_size >= 5:
                    MDL_max_first, t_cut_first = MDL_gaussian_1th(
                        tmp_case_values)

                    MDL_first_scores_cases.append(MDL_max_first)
                    threshold_tmp_first = threshod_F_MDL(
                        delta_first, dimension, window_size)
                    if threshold_tmp_first < 0:
                        threshold_tmp_first = 1e-12
                    # print('frist-orde change: date:{0}, threshold:{1}, MDL score:{2}'.format(
                    #    date_tmp, threshold_tmp_first, MDL_max_first))

                    if MDL_max_first >= threshold_tmp_first:
                        dict_first_changes_cases[
                            dates_tmp_cases[-1]] = MDL_max_first
                        dict_first_windows_cases[
                            dates_tmp_cases[-1]] = t_cut_first
                        if country in event_dict:
                            if date_tmp < event_dict[country]:
                                num_changes_before_event += 1
                                num_1st_changes_before_event += 1

                else:
                    MDL_first_scores_cases.append(0)
                #### end of first order ####

                #### second order ####
                if window_size >= 6:
                    MDL_max_second, t_cut_second = MDL_gaussian_2nd(
                        tmp_case_values)
                    MDL_second_scores_cases.append(MDL_max_second)

                    threshold_tmp_second = threshod_S_MDL(
                        delta_second, dimension, window_size)
                    if threshold_tmp_second < 0:
                        threshold_tmp_second = 1e-12
                    # print('second-orde change: date:{0}, threshold:{1}, MDL score:{2}'.format(
                    #    date_tmp, threshold_tmp_second, MDL_max_second))

                    if MDL_max_second >= threshold_tmp_second:
                        dict_second_changes_cases[
                            dates_tmp_cases[-1]] = MDL_max_second
                        dict_second_windows_cases[
                            dates_tmp_cases[-1]] = t_cut_second
                        if country in event_dict:
                            if date_tmp < event_dict[country]:
                                num_changes_before_event += 1
                                num_2nd_changes_before_event += 1

                else:
                    MDL_second_scores_cases.append(0)
                #### end of second order ####

                window_size += 1
                if MDL_max >= threshold_tmp:
                    dict_changes_cases[dates_tmp_cases[-1]] = MDL_max
                    dic_windows_cases[dates_tmp_cases[-1]] = window_size

                    mean_before = np.mean(tmp_case_values[:cut_point])
                    mean_after = np.mean(tmp_case_values[cut_point:])
                    if mean_before < mean_after:
                        flag_first = False
                        flag_second = False

                        candidate_num += 1
                        if window_size >= 6:
                            first_candidate_num += 1
                            for candidate in t_tmp:
                                if candidate in dict_first_changes_cases:
                                    days_tmp_first = (
                                        dates_tmp_cases[-1].astype(datetime) - candidate.astype(datetime)).days
                                    days_between_zero_first.append(
                                        days_tmp_first)
                                    days_between_zero_earliest.append(
                                        days_tmp_first)
                                    first_detected_num += 1
                                    flag_first = True
                                    break

                        if window_size >= 7:
                            second_candidate_num += 1
                            for candidate in t_tmp:
                                if candidate in dict_second_changes_cases:
                                    days_tmp_second = (
                                        dates_tmp_cases[-1].astype(datetime) - candidate.astype(datetime)).days
                                    days_between_zero_second.append(
                                        days_tmp_second)
                                    if flag_first and days_tmp_second > days_tmp_first:
                                        days_between_zero_earliest = days_between_zero_earliest[
                                            :-1]
                                        days_between_zero_earliest.append(
                                            days_tmp_second)
                                    if not flag_first:
                                        days_between_zero_earliest.append(
                                            days_tmp_second)
                                    second_detected_num += 1
                                    flag_second = True
                                    break

                        if flag_first or flag_second:
                            detected_num += 1

                        flag_first = False
                        flag_second = False

                    window_size = 1
                    # for the change point to have to window sizes
                    dates_tmp_cases.append(dates_tmp_cases[-1])
                    # for the change point to have to window sizes
                    MDL_scores_cases.append(MDL_max)
                    windows_cases.append(window_size - 1)

                    # statistics
                    if country in event_dict:
                        if date_tmp < event_dict[country]:
                            num_changes_before_event += 1
                            num_0th_changes_before_event += 1

                        if country not in decrease_dict:
                            mean_before = np.mean(tmp_case_values[:cut_point])
                            mean_after = np.mean(tmp_case_values[cut_point:])
                            if max_date < date_tmp and date_tmp > event_dict[country] and mean_before > mean_after:
                                decrease_dict[country] = date_tmp
                                #print('data:{}, mean before:{}, mean after:{}'.format(date_tmp, mean_before, mean_after))
                    else:
                        if country not in decrease_no_event_dict:
                            mean_before = np.mean(tmp_case_values[:cut_point])
                            mean_after = np.mean(tmp_case_values[cut_point:])
                            if max_date < date_tmp and mean_before > mean_after:
                                decrease_no_event_dict[country] = date_tmp

            else:
                # print('date:{0}, window size:{1}, MDL score:{2}'.format(
                #    date_tmp, window_size, 0))
                MDL_scores_cases.append(0)
                windows_cases.append(window_size)
                window_size += 1

                MDL_first_scores_cases.append(0)
                MDL_second_scores_cases.append(0)

    if country in event_dict:
        num_change_dict[country] = num_changes_before_event
        num_0th_change_dict[country] = num_0th_changes_before_event
        num_1st_change_dict[country] = num_1st_changes_before_event
        num_2nd_change_dict[country] = num_2nd_changes_before_event

    # zero figure
    # Some change scores at the beginning of the detection may be infinity because
    # the standard deviation is zero due to the same number of daily new
    # cases, e.g., 1
    max_mdl = 0
    for tmp_mdl in MDL_scores_cases:
        if tmp_mdl > max_mdl and tmp_mdl != float('inf'):
            max_mdl = tmp_mdl

    MDL_scores_cases = [x if x != float(
        'inf') else max_mdl for x in MDL_scores_cases]
    MDL_scores_cases[:] /= max_mdl

    for key in dict_changes_cases:
        if dict_changes_cases[key] == float('inf'):
            dict_changes_cases[key] = max_mdl
        dict_changes_cases[key] = dict_changes_cases[key] / max_mdl

    plt.clf()
    plt.figure(figsize=(28, 10))
    plt.rc('font', size=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    # replace X-DATES and Y-VALUES with the variable names
    plt.plot(dates_tmp_cases, MDL_scores_cases)
    plt.gcf().autofmt_xdate()
    if country == 'South_Korea':
        plt.title('South Korea')
    else:
        plt.title(country)
    plt.ylabel('0th D-MDL change score')

    change_index = 1
    for key in dict_changes_cases:
        plt.vlines(key, ymin=0, ymax=dict_changes_cases[
                   key], color='r', linestyle='--')
        change_index += 1

    if country in event_dict:
        plt.vlines(dates_events, ymin=0, ymax=1.1,
                   color='black', linestyle='-', linewidth=2)
        plt.text(dates_events, 0.8, 'Social distancing', color='black')

    plt.ylim(0, 1.1)
    plt.tight_layout()
    if country == 'South Korea' or country == 'United Kingdom' or country == 'United States':
        plt.savefig(output_path + country.replace(' ', '_') + '_zero.png')
        plt.savefig(output_path + country.replace(' ', '_') + '_zero.eps')
    else:
        plt.savefig(output_path + country + '_0_score.png')
        plt.savefig(output_path + country + '_0_score.eps')

    # window figure
    plt.clf()
    plt.figure(figsize=(28, 10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.plot(dates_tmp_cases, windows_cases)
    plt.gcf().autofmt_xdate()
    if country == 'South_Korea':
        plt.title('South Korea')
    else:
        plt.title(country)
    plt.ylabel('Window size')

    max_window_size = max(40, np.nanmax(np.array(windows_cases)) * 1.1)

    for key in dic_windows_cases:
        plt.vlines(key, ymin=2, ymax=max_window_size, color='r', linestyle='-')

    if country in event_dict:
        plt.vlines(dates_events, ymin=0, ymax=max_window_size,
                   color='black', linestyle='-', linewidth=2)
        plt.text(dates_events, 25.1, 'Social distancing', color='black')

    plt.rc('font', size=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.ylim(0, max_window_size)
    plt.tight_layout()
    if country == 'South Korea' or country == 'United Kingdom' or country == 'United States':
        plt.savefig(output_path + country.replace(' ', '_') + '_window.png')
        plt.savefig(output_path + country.replace(' ', '_') + '_window.eps')
    else:
        plt.savefig(output_path + country + '_window_size.png')
        plt.savefig(output_path + country + '_window_size.eps')

    # first order figure
    max_first_mdl = 0
    for tmp_mdl in MDL_first_scores_cases:
        if tmp_mdl > max_first_mdl and tmp_mdl != float('inf'):
            max_first_mdl = tmp_mdl

    MDL_first_scores_cases = [x if x != float(
        'inf') else max_first_mdl for x in MDL_first_scores_cases]

    #MDL_first_max = np.max(MDL_first_scores_cases)
    MDL_first_scores_cases[:] /= max_first_mdl

    for key in dict_first_changes_cases:
        if dict_first_changes_cases[key] == float('inf'):
            dict_first_changes_cases[key] = max_first_mdl
        dict_first_changes_cases[
            key] = dict_first_changes_cases[key] / max_first_mdl

    plt.clf()
    plt.figure(figsize=(28, 10))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.plot(dates_tmp_cases_first, MDL_first_scores_cases)
    plt.gcf().autofmt_xdate()
    if country == 'South_Korea':
        plt.title('South Korea')
    else:
        plt.title(country)
    plt.ylabel('1st D-MDL change score')

    change_index = 1
    for key in dict_first_changes_cases:
        plt.vlines(key, ymin=0, ymax=dict_first_changes_cases[
                   key], color='r', linestyle='--')
        change_index += 1

    if country in event_dict:
        plt.vlines(dates_events, ymin=0, ymax=1.1,
                   color='black', linestyle='-', linewidth=2)
        plt.text(dates_events, 0.8, 'Social distancing', color='black')

    plt.ylim(0, 1.1)
    plt.rc('font', size=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.tight_layout()
    if country == 'South Korea' or country == 'United Kingdom' or country == 'United States':
        plt.savefig(output_path + country.replace(' ', '_') + '_first.png')
        plt.savefig(output_path + country.replace(' ', '_') + '_first.eps')
    else:
        plt.savefig(output_path + country + '_1_score.png')
        plt.savefig(output_path + country + '_1_score.eps')

    # second order figure
    max_second_mdl = 0
    for tmp_mdl in MDL_second_scores_cases:
        if tmp_mdl > max_second_mdl and tmp_mdl != float('inf'):
            max_second_mdl = tmp_mdl

    MDL_second_scores_cases = [x if x != float(
        'inf') else max_second_mdl for x in MDL_second_scores_cases]

    #MDL_first_max = np.max(MDL_first_scores_cases)
    MDL_second_scores_cases[:] /= max_second_mdl

    for key in dict_second_changes_cases:
        if dict_second_changes_cases[key] == float('inf'):
            dict_second_changes_cases[key] = max_second_mdl
        dict_second_changes_cases[
            key] = dict_second_changes_cases[key] / max_second_mdl

    plt.clf()
    plt.figure(figsize=(28, 10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.plot(dates_tmp_cases_second, MDL_second_scores_cases)
    plt.gcf().autofmt_xdate()
    if country == 'South_Korea':
        plt.title('South Korea')
    else:
        plt.title(country)
    plt.ylabel('2nd D-MDL change score')

    change_index = 1
    for key in dict_second_changes_cases:
        plt.vlines(key, ymin=0, ymax=dict_second_changes_cases[
                   key], color='r', linestyle='--')
        #plt.text(key, dict_second_changes_cases[key]+0.025, str(change_index), color='black', bbox=box)
        change_index += 1

    if country in event_dict:
        plt.vlines(dates_events, ymin=0, ymax=1.1,
                   color='black', linestyle='-', linewidth=2)
        plt.text(dates_events, 0.8, 'Social distancing', color='black')

    plt.ylim(0, 1.1)
    plt.rc('font', size=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.tight_layout()
    if country == 'South Korea' or country == 'United Kingdom' or country == 'United States':
        plt.savefig(output_path + country.replace(' ', '_') + '_second.png')
        plt.savefig(output_path + country.replace(' ', '_') + '_second.eps')
    else:
        plt.savefig(output_path + country + '_2_score.png')
        plt.savefig(output_path + country + '_2_score.eps')

    # cases figure
    plt.clf()
    plt.figure(figsize=(28, 10))
    case_day = []
    days = []
    for key in case_values:
        case_day.append(case_values[key])
        days.append(key)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    plt.plot(days, case_day)
    plt.gcf().autofmt_xdate()
    plt.ylabel('Cases')
    if country == 'South_Korea':
        plt.title('South Korea')
    else:
        plt.title(country)

    if country in event_dict:
        plt.vlines(dates_events, ymin=0, ymax=np.max(case_day) *
                   1.1, color='black', linestyle='-', linewidth=2)
        plt.text(dates_events, np.max(case_day) * 0.8,
                 'Social distancing', color='black')

    plt.ylim(0, np.max(case_day) * 1.1)
    plt.rc('font', size=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.tight_layout()
    plt.savefig(output_path + country + '_case.png')
    plt.savefig(output_path + country + '_case.eps')

    # log for posts to slack
    if np.datetime64(max_df_date) in dict_changes_cases:
        zeroth_alarm = 1
    else:
        zeroth_alarm = 0

    if np.datetime64(max_df_date) in dict_first_changes_cases:
        first_alarm = 1
    else:
        first_alarm = 0

    if np.datetime64(max_df_date) in dict_second_changes_cases:
        second_alarm = 1
    else:
        second_alarm = 0

    zeroth_alarm_dates = list(dict_changes_cases.keys())
    zeroth_alarm_dates.sort()
    zeroth_alarms_in_stat_interval = len([date for date in zeroth_alarm_dates if stat_date_start <=
                                          date <= stat_date_end])

    first_alarm_dates = list(dict_first_changes_cases.keys())
    first_alarm_dates.sort()
    first_alarms_in_stat_interval = len([date for date in first_alarm_dates if stat_date_start <=
                                         date <= stat_date_end])

    second_alarm_dates = list(dict_second_changes_cases.keys())
    second_alarm_dates.sort()
    second_alarms_in_stat_interval = len([date for date in second_alarm_dates if stat_date_start <=
                                          date <= stat_date_end])

    sum_alarms_in_stat_interval = zeroth_alarms_in_stat_interval + \
        first_alarms_in_stat_interval + second_alarms_in_stat_interval

    if len(zeroth_alarm_dates) == 0:
        window_shrink_latest_date = None
        _window_shrink_latest_date = min_df_date
    else:
        window_shrink_latest_date = zeroth_alarm_dates[
            len(zeroth_alarm_dates) - 1]
        _window_shrink_latest_date = window_shrink_latest_date

    # print('window_shrink_latest_date')
    # print(window_shrink_latest_date)

    first_alarms_in_current_window = [
        date for date in first_alarm_dates if date > _window_shrink_latest_date]
    second_alarms_in_current_window = [
        date for date in second_alarm_dates if date > _window_shrink_latest_date]

    if len(first_alarms_in_current_window) == 0:
        first_latest_alarm_date = None
    else:
        first_latest_alarm_date = first_alarms_in_current_window[
            len(first_alarms_in_current_window) - 1]

    if len(second_alarms_in_current_window) == 0:
        second_latest_alarm_date = None
    else:
        second_latest_alarm_date = second_alarms_in_current_window[
            len(second_alarms_in_current_window) - 1]

    # print('first_latest_alarm_date')
    # print(first_latest_alarm_date)
    # print('second_latest_alarm_date')
    # print(second_latest_alarm_date)

    if first_latest_alarm_date is None and second_latest_alarm_date is None:
        sign_latest_date = None
        sign_oldest_date = None
    elif first_latest_alarm_date is None:
        sign_latest_date = pd.to_datetime(second_latest_alarm_date, format='%Y-%m-%d')
        sign_oldest_date = pd.to_datetime(second_latest_alarm_date, format='%Y-%m-%d')
    elif second_latest_alarm_date is None:
        sign_latest_date = pd.to_datetime(first_latest_alarm_date, format='%Y-%m-%d')
        sign_oldest_date = pd.to_datetime(first_latest_alarm_date, format='%Y-%m-%d')
    else:
        sign_latest_date = pd.to_datetime(max(
            first_latest_alarm_date, second_latest_alarm_date), format='%Y-%m-%d')
        sign_oldest_date = pd.to_datetime(min(
            first_latest_alarm_date, second_latest_alarm_date), format='%Y-%m-%d')

    if sign_oldest_date is None:
        predicted_window_shrink_date = None
    else:
        predicted_window_shrink_date = pd.to_datetime(
            sign_oldest_date + np.timedelta64(6, 'D'), format='%Y-%m-%d')

    row_alarms = pd.DataFrame({'CountryAndTerritory': [country],
                               'data_latest_date': max_df_date,
                               'cases': case_values[np.datetime64(max_df_date)],
                               '0th': zeroth_alarm,
                               '1st': first_alarm,
                               '2nd': second_alarm,
                               '0th_alarms': zeroth_alarms_in_stat_interval,
                               '1st_alarms': first_alarms_in_stat_interval,
                               '2nd_alarms': second_alarms_in_stat_interval,
                               'sum_alarms': sum_alarms_in_stat_interval,
                               'window_shrink_latest_date': window_shrink_latest_date,
                               'sign_latest_date': sign_latest_date,
                               'predicted_window_shrink_date': predicted_window_shrink_date
                               })
    alarms_for_each_country = pd.concat(
        [alarms_for_each_country, row_alarms], axis=0)

alarms_for_each_country['sign_latest_date']=pd.to_datetime(
    alarms_for_each_country['sign_latest_date'], format='%Y-%m-%d')
alarms_for_each_country['predicted_window_shrink_date']=pd.to_datetime(
    alarms_for_each_country['predicted_window_shrink_date'], format='%Y-%m-%d')
alarms_for_each_country.to_csv('data/gaussian_alarm_results.csv', index=False)

# Number of alarms for non-downward countries
num_change_decrease = []
num_0th_change_decrease = []
num_1st_change_decrease = []
num_2nd_change_decrease = []
num_change_increase = []
num_0th_change_increase = []
num_1st_change_increase = []
num_2nd_change_increase = []
num_0th_days_before_event_decrease = []
num_1st_days_before_event_decrease = []
num_2nd_days_before_event_decrease = []
num_0th_days_before_event_increase = []
num_1st_days_before_event_increase = []
num_2nd_days_before_event_increase = []
num_days_event_to_decrease = []
num_days_increase_to_decrease_event = []
num_days_increase_to_decrease_no_event = []
for key in candidates_country:
    if key in event_dict:
        if key in decrease_dict:
            print('country:{0}, change num:{1}, event-decrease:{2}'.format(key, num_change_dict[
                  key], (decrease_dict[key].astype(datetime) - event_dict[key]).days))
            num_change_decrease.append(num_change_dict[key])
            if key in num_0th_change_dict:
                num_0th_change_decrease.append(num_0th_change_dict[key])
            else:
                num_0th_change_decrease.append(0)

            if key in num_1st_change_dict:
                num_1st_change_decrease.append(num_1st_change_dict[key])
            else:
                num_1st_change_decrease.append(0)

            if key in num_2nd_change_dict:
                num_2nd_change_decrease.append(num_2nd_change_dict[key])
            else:
                num_2nd_change_decrease.append(0)

            num_days_event_to_decrease.append(
                (decrease_dict[key].astype(datetime) - event_dict[key]).days)
        else:
            num_change_increase.append(num_change_dict[key])
            if key in num_0th_change_dict:
                num_0th_change_increase.append(num_0th_change_dict[key])
            else:
                num_0th_change_increase.append(0)

            if key in num_1st_change_dict:
                num_1st_change_increase.append(num_1st_change_dict[key])
            else:
                num_1st_change_increase.append(0)

            if key in num_2nd_change_dict:
                num_2nd_change_increase.append(num_2nd_change_dict[key])
            else:
                num_2nd_change_increase.append(0)

# Number of alarms for non-downward countries
for key in event_dict:
    if key not in decrease_dict:
        print('country:{0}, change num:{1}'.format(key, num_change_dict[key]))

# total number of alarms before the event for downward countries
np.mean(num_change_decrease), np.std(num_change_decrease)

# total number of alarms before the event for non-downward countries
np.mean(num_change_increase), np.std(num_change_increase)

# number of 0th alarms before the event for downward countries and
# non-downward ones, respectively
np.mean(num_0th_change_decrease), np.std(num_0th_change_decrease), np.mean(
    num_0th_change_increase), np.std(num_0th_change_increase)

# number of 1th alarms before the event for downward countries and
# non-downward ones, respectively
np.mean(num_1st_change_decrease), np.std(num_1st_change_decrease), np.mean(
    num_1st_change_increase), np.std(num_1st_change_increase)

# number of 2nd alarms before the event for downward countries and
# non-downward ones, respectively
np.mean(num_2nd_change_decrease), np.std(num_2nd_change_decrease), np.mean(
    num_2nd_change_increase), np.std(num_2nd_change_increase)

plt.clf()
plt.figure(figsize=(9, 6))
bins = np.arange(0.5, 11.5, 1)  # fixed bin size
plt.xlim([min(num_change_decrease) - 1, max(num_change_decrease) + 1])
plt.hist(num_change_decrease, bins=bins, rwidth=0.422)
plt.ylabel('# Countries')
plt.xlabel('# Change points')
plt.title('Downward')
plt.rc('font', size=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.tight_layout()
plt.savefig(output_path + 'decrease_change_before_event.png')
plt.savefig(output_path + 'decrease_change_before_event.eps')

plt.clf()
plt.figure(figsize=(11, 6))
plt.xlim([min(num_change_increase) - 1, max(num_change_increase) + 1])
bins = np.arange(-0.5, 19.5, 1)
plt.hist(num_change_increase, bins=bins, rwidth=0.47)
plt.ylabel('# Countries')
plt.xlabel('# Change points')
plt.title('Non-downward')
plt.rc('font', size=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.tight_layout()
plt.savefig(output_path + 'increase_change_before_event.png')
plt.savefig(output_path + 'increase_change_before_event.eps')

# total number of outbreaks and number of outbreaks whose signs were
# detected by either the 1st or the 2nd D-MDL
candidate_num, detected_num

# Total number of outbreaks that allowed for the 1st D-MDL sign detection
# and Number of outbreaks whose signs were detected by the 1st D-MDL
first_candidate_num, first_detected_num

# Total number of outbreaks that allowed for the 2nd D-MDL sign detection
# and Number of outbreaks whose signs were detected by the 2nd D-MDL
second_candidate_num, second_detected_num

# Number of days before an outbreak for the first sign of either the 1st
# or the 2nd D-MDL
np.mean(days_between_zero_earliest), np.std(days_between_zero_earliest), np.min(
    days_between_zero_earliest), np.max(days_between_zero_earliest)

# Number of days before an outbreak for the first sign of the 1st
np.mean(days_between_zero_first), np.std(days_between_zero_first), np.min(
    days_between_zero_first), np.max(days_between_zero_first)

# Number of days before an outbreak for the first sign of the 2nd
np.mean(days_between_zero_second), np.std(days_between_zero_second), np.min(
    days_between_zero_second), np.max(days_between_zero_second)

plt.clf()
plt.figure(figsize=(14, 8))
bins = np.arange(0.5, 23.5, 1)  # fixed bin size
plt.xlim([min(days_between_zero_earliest) - 1,
          max(days_between_zero_earliest) + 1])
plt.hist(days_between_zero_earliest, bins=bins, rwidth=0.446)
plt.ylabel('# Outbreaks')
plt.xlabel('# Days')
plt.rc('font', size=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.tight_layout()
plt.savefig(output_path + 'days_between_zero_earliest.png')
plt.savefig(output_path + 'days_between_zero_earliest.eps')

plt.clf()
plt.figure(figsize=(14, 6))
bins = np.arange(0.5, 23.5, 1)  # fixed bin size
plt.xlim([min(days_between_zero_first) - 1, max(days_between_zero_first) + 1])
plt.hist(days_between_zero_first, bins=bins, rwidth=0.446)
plt.ylabel('# Outbreaks')
plt.xlabel('# Days')
plt.title('1st D-MDL')
plt.rc('font', size=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.tight_layout()
plt.savefig(output_path + 'days_between_zero_first.png')
plt.savefig(output_path + 'days_between_zero_first.eps')

plt.clf()
plt.figure(figsize=(11, 6))
bins = np.arange(0.5, 23.5, 1)  # fixed bin size
plt.xlim([min(days_between_zero_second) - 1,
          max(days_between_zero_second) + 1])
plt.hist(days_between_zero_second, bins=bins, rwidth=0.495)
plt.ylabel('# Outbreaks')
plt.xlabel('# Days')
plt.title('2nd D-MDL')
plt.rc('font', size=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.tight_layout()
plt.savefig(output_path + 'days_between_zero_second.png')
plt.savefig(output_path + 'days_between_zero_second.eps')

# Number of days from event’s date to the first downward change’s (0th
# D-MDL alarm)date
np.mean(num_days_event_to_decrease), np.std(
    num_days_event_to_decrease), np.max(num_days_event_to_decrease)

plt.clf()
plt.figure(figsize=(15.5, 6))
bins = np.arange(2.5, 44.5, 1)  # fixed bin size
plt.xlim([min(num_days_event_to_decrease) - 1,
          max(num_days_event_to_decrease) + 1])
# plt.xticks(range(6))
plt.hist(num_days_event_to_decrease, bins=bins, rwidth=0.455)
plt.ylabel('# Countries')
plt.xlabel('# Days')
#plt.title('Days from event to ')
plt.rc('font', size=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.tight_layout()
plt.savefig(output_path + 'days_event_to_decrease.png')
plt.savefig(output_path + 'days_event_to_decrease.eps')
