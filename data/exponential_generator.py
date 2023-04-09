import sys
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import mdlcstat_adwin1
from functools import partial
import pandas as pd
import datetime as dt
from copy import deepcopy
from datetime import timedelta
import requests
import multiprocessing as multi
from multiprocessing import Pool
import scipy.linalg as sl
import config

# data source
#data_url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
filename = 'covid19.csv'
#open(filename, 'wb').write(requests.get(data_url).content)
df_all_origin = pd.read_csv(filename)
df_all = pd.read_csv(filename)
df_all = df_all[["countriesAndTerritories", "dateRep", "cases"]]
df_all["dateRep"] = pd.to_datetime(df_all["dateRep"], format="%d/%m/%Y")
df_all.columns = ["country", "date", "cases"]

output_path = './data/exponential_figs/'


def specify_startingpoint(ctr):
    # Note that the results may be slightly different because the dataset of past dates was updated day-by-day.
    #df_all = pd.read_csv( '../../data/covid_data.csv')
    # df_all=pd.read_csv("./covid19.csv")
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

    for index, row in df_all_origin.iterrows():
        if row['countriesAndTerritories'] in candidates_country:
            # if row['country'] in candidates_country:

            date = dt.datetime(row['year'], row['month'], row['day'])
            df_candidates = df_candidates.append({'country': row['countriesAndTerritories'], 'date': date,
                                                  'cases': row['cases']},
                                                 ignore_index=True)

    t = np.arange(dt.datetime(2020, 1, 1), dt.datetime(2020, 5, 1),
                  dt.timedelta(days=1))  # The timeline for online detection

    t_candidates = []
    t_num = []
    for date_tmp in t:
        row = df_candidates.loc[(df_candidates['country'] == ctr) & (
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

    return pd.to_datetime(t_candidates[0])

'''
def preprocessing(ctr):  # prepare data
    """
    Return a dataframe that contains date and logarithm of cumulative cases in a country.

    parameters:
        ctr: country name

    returns:
        dataframe with date and logarithm of cumulative cases
    """
    #df = pd.read_csv('../../data/covid_data.csv')
    df = df_all[df_all["country"] == ctr][["date", "cases"]]
    #df = df[df["countriesAndTerritories"] == ctr][["dateRep", "cases"]]
    #df["dateRep"] = pd.to_datetime(df["dateRep"], format="%d/%m/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    df.columns = ["date", ctr]

    df[ctr] = df[ctr].cumsum()
    df = df[df[ctr] > 0]
    df = df.reset_index(drop=True)
    df[ctr] = np.log(df[ctr])

    return df
'''


def preprocessing(ctr):  # prepare data
    """
    Return a dataframe that contains date and logarithm of cumulative cases in a country.

    parameters:
        ctr: country name

    returns:
        dataframe with date and logarithm of cumulative cases
    """
    # df=pd.read_csv('../../data/covid_data.csv')
    df = df_all[df_all["country"] == ctr][["date", "cases"]]
    # df=pd.read_csv("./covid19.csv")
    #df = df[df["countriesAndTerritories"] == ctr][["dateRep", "cases"]]
    #df["dateRep"] = pd.to_datetime(df["dateRep"], format="%d/%m/%Y")
    #df = df.sort_values("dateRep").reset_index(drop=True)
    #df.columns = ["date", ctr]

    #df = df[df["countriesAndTerritories"] == ctr][["dateRep", "cases"]]
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    df.columns = ["date", ctr]

    starting_date = specify_startingpoint(ctr)

    # deadline=pd.to_datetime("2020-4-30")
    #df=df[(starting_date<=df["date"]) & (df["date"]<=deadline) ].reset_index(drop=True)
    df = df[(starting_date <= df["date"])].reset_index(drop=True)

    df[ctr][0] = 1
    df[ctr] = df[ctr].cumsum()

    df = df.reset_index(drop=True)
    df[ctr] = np.log(df[ctr])

    return df


def calculate_alpha(params):  # adjust alpha
    """
    Calculate upper-bounds of hypothesis testings given by 1st and 2nd from data.

    parameters:
        params: hyperparameters for calculating LNML of Gaussian

    returns:
        alpha_1, alpha_2
    """
    country = "Japan"
    df = preprocessing(country)
    events = [pd.to_datetime('2020/2/27')]
    # lnml_gaussian = partial(mdlcstat_adwin1.lnml_gaussian,
    #                        sigma_given=params['sigma_given'])
    nml_regression = mdlcstat_adwin1.nml_regression
    # batch_adwin1 = mdlcstat_adwin1.batch(
    # lossfunc=lnml_gaussian, d=2, alpha=1, delta=0.05, how_to_drop='all',
    # preprocess=True)
    batch_adwin1 = mdlcstat_adwin1.batch(
        lossfunc=nml_regression, d=3, alpha=1, delta=0.05, how_to_drop='all', preprocess=False)

    # ret_window, cut, mdl_0, mdl_1, mdl_2, alarm_0_p, alarm_0_m, alarm_1, alarm_2 = batch_adwin1.decision_function(
    #    np.array(df[country]))
    ret_window, cut, mdl_0, mdl_1, mdl_2, alarm_0, alarm_1, alarm_2 = batch_adwin1.decision_function(
        np.array(df[country]))

    event_at = len(df[df['date'] < events[0]])
    dif_1 = 0
    dif_2 = 0

    if np.isnan(mdl_1[event_at]) == False:
        dif_1 = 0
    else:
        abs_dif = 1
        while True:
            stat_p = mdl_1[event_at + abs_dif]
            stat_m = mdl_1[event_at - abs_dif]

            if np.isnan(stat_p) == True and np.isnan(stat_m) == True:
                abs_dif += 1
                continue
            elif np.isnan(stat_p) == False and np.isnan(stat_m) == False:
                if stat_p > stat_m:
                    dif_1 = abs_dif
                    break
                else:
                    dif_1 = -abs_dif
                    break
            else:
                if np.isnan(stat_p) == False:
                    dif_1 = abs_dif
                    break
                else:
                    dif_1 = -abs_dif
                    break

    if np.isnan(mdl_2[event_at]) == False:
        dif_2 = 0
    else:
        abs_dif = 1
        while True:
            stat_p = mdl_2[event_at + abs_dif]
            stat_m = mdl_2[event_at - abs_dif]

            if np.isnan(stat_p) == True and np.isnan(stat_m) == True:  # どっちもnan
                abs_dif += 1
                continue
            elif np.isnan(stat_p) == False and np.isnan(stat_m) == False:
                if stat_p > stat_m:
                    dif_2 = abs_dif
                    break
                else:
                    dif_2 = -abs_dif
                    break
            else:
                if np.isnan(stat_p) == False:
                    dif_2 = abs_dif
                    break
                else:
                    dif_2 = -abs_dif
                    break

    if ret_window[event_at + dif_1] == 0:
        win_1 = ret_window[event_at + dif_1 - 1] + 1
    else:
        win_1 = ret_window[event_at + dif_1]

    if ret_window[event_at + dif_2] == 0:
        win_2 = ret_window[event_at + dif_2 - 1] + 1
    else:
        win_2 = ret_window[event_at + dif_2]

    alpha_1 = np.exp(2 * np.log(win_1 / 2) - mdl_1[event_at])
    alpha_2 = np.exp(2 * np.log(win_2 / 2) - mdl_2[event_at] / 2)

    # if alpha exceeds 1, set them 0.99.
    if alpha_1 > 1:
        alpha_1 = 0.99
    if alpha_2 > 1:
        alpha_2 = 0.99

    return alpha_1, alpha_2


def separate_changepoints(data, alarm, cut):
    alarm_p = []
    alarm_m = []

    for i in range(len(alarm)):
        if i == 0:
            data_before = data[0:cut[i]]
            data_after = data[cut[i]:alarm[i] + 1]
        else:
            data_before = data[alarm[i - 1] + 1:cut[i]]
            data_after = data[cut[i]:alarm[i] + 1]

        data_before = np.matrix(data_before).T
        n = data_before.shape[0]
        W = np.ones((2, n))
        W[1, :] = np.arange(1, n + 1)
        beta_before = sl.pinv(W.dot(W.T)).dot(W).dot(data_before)

        data_after = np.matrix(data_after).T
        n = data_after.shape[0]
        W = np.ones((2, n))
        W[1, :] = np.arange(1, n + 1)
        beta_after = sl.pinv(W.dot(W.T)).dot(W).dot(data_after)

        beta_before = np.squeeze(np.asarray(beta_before))
        beta_after = np.squeeze(np.asarray(beta_after))

        if beta_before[1] < beta_after[1]:
            alarm_p.append(alarm[i])
        else:
            alarm_m.append(alarm[i])

    # print(alarm)
    # print(alarm_p)
    # print(alarm_m)

    return np.array(alarm_p), np.array(alarm_m)


def _country_stat(country, events_all, params):
    """
    Plot a graphs for each country.
    parameters:
        country: country name
        events: date of events for enforcement of social distancing
        params: hyperparameters for calculating statistics
    """

    # グラフのメモリの表示期間を調整する
    # interval = 62

    if country == "Korea, South":
        ctr = "South_Korea"
        printctr = "South Korea"
    elif country == "US":
        ctr = "United_States_of_America"
        printctr = "United States of America"
    elif country == "Saudi Arabia":
        ctr = "Saudi_Arabia"
        printctr = "Saudi Arabia"
    elif country == "United Arab Emirates":
        ctr = "United_Arab_Emirates"
        printctr = "United Arab Emirates"
    elif country == "United Kingdom":
        ctr = "United_Kingdom"
        printctr = "United Kingdom"
    else:
        ctr = country
        printctr = country

    events = events_all[country]

    # data preprocessing
    df = preprocessing(ctr)

    start = df['date'][0]
    end = df['date'][len(df['date']) - 1]

    alpha_1 = params["alpha_1"]
    alpha_2 = params["alpha_2"]

    nml_regression = mdlcstat_adwin1.nml_regression
    batch_adwin1 = mdlcstat_adwin1.batch(
        lossfunc=nml_regression, d=3, alpha=alpha_1, delta=0.05, how_to_drop='all', preprocess=False)
    ret_window, cut, mdl_0, mdl_1, _, alarm_0, alarm_1, _ = batch_adwin1.decision_function(
        np.array(df[ctr]))
    batch_adwin1 = mdlcstat_adwin1.batch(
        lossfunc=nml_regression, d=3, alpha=alpha_2, delta=0.05, how_to_drop='all', preprocess=False)
    _, _, _, _, mdl_2, _, _, alarm_2 = batch_adwin1.decision_function(
        np.array(df[ctr]))

    mdl_1 = np.nan_to_num(mdl_1)
    mdl_2 = np.nan_to_num(mdl_2)

    #alarm_0 = np.r_[alarm_0_p, alarm_0_m]
    alarm_0 = np.sort(alarm_0)

    cut = cut[np.where(cut != -1)]
    for i in range(len(cut)):
        if i != 0:
            cut[i] += alarm_0[i - 1] + 1

    alarm_0_index = deepcopy(alarm_0)
    alarm_1_index = deepcopy(alarm_1)
    alarm_2_index = deepcopy(alarm_2)

    # separate change points
    alarm_0_p, alarm_0_m = separate_changepoints(
        np.array(df[ctr]), alarm_0, cut)

    alarm_0 = df['date'][alarm_0].reset_index(drop=True)
    #alarm_0_p = df['date'][alarm_0_p].reset_index(drop=True)
    #alarm_0_m = df['date'][alarm_0_m].reset_index(drop=True)
    alarm_1 = df['date'][alarm_1].reset_index(drop=True)
    alarm_2 = df['date'][alarm_2].reset_index(drop=True)
    cut = df['date'][cut].reset_index(drop=True)

    # Linchuanの調整
    cumcases = np.exp(np.copy(np.array(df[ctr])))
    local_cumcases = np.exp(np.copy(np.array(df[ctr])))
    localcum_supplementary = []

    for i in range(len(alarm_0_index)):
        if i == 0:
            localcum_supplementary.append(cumcases[alarm_0_index[i]])
        else:
            localcum_supplementary.append(
                cumcases[alarm_0_index[i]] - cumcases[alarm_0_index[i - 1]])

        if alarm_0_index[i] + 1 != len(cumcases):
            local_cumcases[alarm_0_index[i]:] = cumcases[
                alarm_0_index[i]:] - cumcases[alarm_0_index[i]] + 1
            # print(df_cumcases)

    localcum_supplementary = np.array(localcum_supplementary)

    df_localcum = pd.DataFrame(
        {'date': df['date'].values, 'local_cumcases': local_cumcases})
    df_supplementary = pd.DataFrame(
        {'date': df['date'][alarm_0_index].values, 'local_cumcases': localcum_supplementary})
    df_concat_localcum = pd.concat([df_localcum, df_supplementary], axis=0)
    df_concat_localcum = df_concat_localcum.sort_values(
        ['date', 'local_cumcases'], ascending=[False, True])

    print('date for alarms of 0th D-MDL')
    print(alarm_0)
    print('cutpoint')
    print(cut)

    df_window = pd.DataFrame({'date': df['date'].values, 'window': ret_window})
    df_supplementary = pd.DataFrame({'date': df['date'][
                                    alarm_0_index].values, 'window': ret_window[alarm_0_index - 1] + 1})
    df_concat = pd.concat([df_window, df_supplementary], axis=0)
    df_concat = df_concat.sort_values(
        ['date', 'window'], ascending=[False, True])

    if len(alarm_0_index) != 0 and df['date'][alarm_0_index[-1]] == end:
        zeroth_alarm = 1
    else:
        zeroth_alarm = 0

    if len(alarm_1_index) != 0 and df['date'][alarm_1_index[-1]] == end:
        first_alarm = 1
    else:
        first_alarm = 0

    if len(alarm_2_index) != 0 and df['date'][alarm_2_index[-1]] == end:
        second_alarm = 1
    else:
        second_alarm = 0

    raw_alarms = pd.DataFrame({'CountryAndTerritory': [ctr], 'data_latest_date': end,
                               'cases': df_all[(df_all["country"] == ctr) & (df_all["date"] == end)]["cases"],
                               '0th': zeroth_alarm, '1st': first_alarm, '2nd': second_alarm})
    print(raw_alarms)

    return df, df_concat_localcum, mdl_0, mdl_1, mdl_2, alarm_0_index, alarm_1_index, alarm_2_index, alarm_0_p, alarm_0_m, df_concat, printctr, ctr, events, output_path, ret_window, raw_alarms


def country_graph(df, df_concat_localcum, mdl_0, mdl_1, mdl_2, alarm_0_index, alarm_1_index, alarm_2_index,
                  alarm_0_p, alarm_0_m, df_concat, printctr, ctr, events, output_path, ret_window):
    # plot data
    # 0th D-MDL
    plt.clf()
    plt.rc('font', size=36)
    plt.rc('xtick', labelsize=36)
    plt.rc('ytick', labelsize=36)
    plt.figure(figsize=(28, 10))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=config.INTERVAL_LONG))

    # dates in datetime format and change scores
    plt.plot(df['date'], mdl_0 / np.nanmax(mdl_0))
    plt.gcf().autofmt_xdate()
    plt.title(printctr)
    plt.ylabel('0th D-MDL change score')

    # for marking changes
    for p_index in alarm_0_index:
        plt.vlines(df['date'][p_index], ymin=0, ymax=mdl_0[
                   p_index] / np.nanmax(mdl_0), color='r', linestyle='--')

    # for making social distancing event
    plt.vlines(events, ymin=0, ymax=1.1, color='black',
               linestyle='-', linewidth=2)
    plt.text(events, 0.8, 'Social distancing', color='black')

    #plt.xlim(start - dt.timedelta(days=3), end + dt.timedelta(days=3))
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path + ctr + '_0_score.png')
    plt.savefig(output_path + ctr + '_0_score.eps')

    # 1st D-MDL
    plt.clf()
    plt.rc('font', size=36)
    plt.rc('xtick', labelsize=36)
    plt.rc('ytick', labelsize=36)
    plt.figure(figsize=(28, 10))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=config.INTERVAL_LONG))
    # dates in datetime format and change scores
    plt.plot(df['date'], mdl_1 / np.nanmax(mdl_1))
    plt.gcf().autofmt_xdate()
    plt.title(printctr)
    plt.ylabel('1st D-MDL change score')

    # for marking signs of changes
    for p_index in alarm_1_index:
        plt.vlines(df['date'][p_index], ymin=0, ymax=mdl_1[
                   p_index] / np.nanmax(mdl_1), color='r', linestyle='--')

    # for making social distancing event
    plt.vlines(events, ymin=0, ymax=1.1, color='black',
               linestyle='-', linewidth=2)
    plt.text(events, 0.8, 'Social distancing', color='black')

    #plt.xlim(start - dt.timedelta(days=3), end + dt.timedelta(days=3))
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path + ctr + '_1_score.png')
    plt.savefig(output_path + ctr + '_1_score.eps')

    # 2nd D-MDL
    plt.rc('font', size=36)
    plt.rc('xtick', labelsize=36)
    plt.rc('ytick', labelsize=36)
    plt.clf()
    plt.figure(figsize=(28, 10))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=config.INTERVAL_LONG))
    # dates in datetime format and change scores
    plt.plot(df['date'], mdl_2 / np.nanmax(mdl_2))
    plt.gcf().autofmt_xdate()
    plt.title(printctr)
    plt.ylabel('2nd D-MDL change score')

    # for marking signs of changes
    for p_index in alarm_2_index:
        plt.vlines(df['date'][p_index], ymin=0, ymax=mdl_2[
                   p_index] / np.nanmax(mdl_2), color='r', linestyle='--')

    # for making social distancing event
    plt.vlines(events, ymin=0, ymax=1.1, color='black',
               linestyle='-', linewidth=2)
    plt.text(events, 0.8, 'Social distancing', color='black')

    #plt.xlim(start - dt.timedelta(days=3), end + dt.timedelta(days=3))
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path + ctr + '_2_score.png')
    plt.savefig(output_path + ctr + '_2_score.eps')

    # cases
    plt.rc('font', size=36)
    plt.rc('xtick', labelsize=36)
    plt.rc('ytick', labelsize=36)
    plt.clf()
    plt.figure(figsize=(28, 10))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=config.INTERVAL_LONG))
    # dates in datetime format and change scores
    #plt.plot(df['date'], np.exp(df[ctr]))
    # df_concat_localcum["local_cumcases"][0]=1
    # dates in datetime format and change scores
    plt.plot(df_concat_localcum["date"], df_concat_localcum["local_cumcases"])

    plt.gcf().autofmt_xdate()
    plt.yscale('log')
    plt.title(printctr)
    plt.ylabel('Local cumulative cases')

    # for making social distancing event
    plt.vlines(events, ymin=0, ymax=max(
        np.exp(df[ctr]))**1.1, color='black', linestyle='-', linewidth=2)
    plt.text(events, np.nanmax(np.exp(df[ctr]))
             ** 0.8, 'Social distancing', color='black')

    for p_index in alarm_0_p:
        plt.vlines(df['date'][p_index], ymin=0, ymax=max(
            np.exp(df[ctr]))**1.1, color='b', linestyle='-')

    for p_index in alarm_0_m:
        plt.vlines(df['date'][p_index], ymin=0, ymax=max(
            np.exp(df[ctr]))**1.1, color='r', linestyle='-')

    #plt.xlim(start - dt.timedelta(days=3), end + dt.timedelta(days=3))
    plt.ylim(0, max(np.exp(df[ctr])) ** 1.1)
    plt.tight_layout()
    plt.savefig(output_path + ctr + '_case.png')
    plt.savefig(output_path + ctr + '_case.eps')

    # window size
    plt.rc('font', size=36)
    plt.rc('xtick', labelsize=36)
    plt.rc('ytick', labelsize=36)
    plt.clf()
    plt.figure(figsize=(28, 10))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=config.INTERVAL_LONG))
    plt.plot(df_concat['date'], df_concat['window'])
    plt.gcf().autofmt_xdate()
    plt.title(printctr)
    plt.ylabel('Window size')

    # for marking changes
    for p_index in alarm_0_p:
        plt.vlines(df['date'][p_index], ymin=0, ymax=max(
            40, np.nanmax(ret_window) * 1.1) * 1.1, color='b', linestyle='-')

    for p_index in alarm_0_m:
        plt.vlines(df['date'][p_index], ymin=0, ymax=max(
            40, np.nanmax(ret_window) * 1.1) * 1.1, color='r', linestyle='-')

    # for making social distancing event
    plt.vlines(events, ymin=0, ymax=max(40, np.nanmax(ret_window) * 1.1)
               * 1.1, color='black', linestyle='-', linewidth=2)
    plt.text(events, 0.8 * max(40, np.nanmax(ret_window) * 1.1),
             'Social distancing', color='black')

    #plt.xlim(start - dt.timedelta(days=3), end + dt.timedelta(days=3))
    plt.ylim(0, max(40, np.nanmax(ret_window) * 1.1))
    plt.tight_layout()
    plt.savefig(output_path + ctr + '_window_size.png')
    plt.savefig(output_path + ctr + '_window_size.eps')


countries = ["Austria", "Belarus", "Belgium", "Brazil", "Canada", "Chile", "China", "Ecuador",
             "France", "Germany", "India", "Indonesia", "Iran",  "Ireland", "Israel", "Italy", "Japan", "Mexico",
             "Netherlands", "Pakistan", "Peru", "Poland", "Portugal", "Qatar", "Romania", "Russia",
             "Saudi Arabia", "Singapore", "Korea, South", "Spain", "Sweden", "Switzerland", "Turkey",
             "United Arab Emirates", "Ukraine", "United Kingdom", "US", "Tokyo"]

#countries = ["Japan", "Tokyo", "Belgium", "Brazil"]

events_Austria = [pd.to_datetime('2020/3/16')]
events_Belarus = [pd.to_datetime('2020/4/9')]
events_Belgium = [pd.to_datetime('2020/3/18')]
events_Brazil = [pd.to_datetime('2020/3/24')]
events_Canada = [pd.to_datetime('2020/3/17')]
events_Chile = [pd.to_datetime('2020/3/26')]
events_China = [pd.to_datetime('2020/1/23')]
events_Ecuador = [pd.to_datetime('2020/3/16')]
events_France = [pd.to_datetime('2020/3/17')]
events_Germany = [pd.to_datetime('2020/3/16')]
events_India = [pd.to_datetime('2020/3/25')]
events_Indonesia = [pd.to_datetime('2020/4/6')]
events_Iran = [pd.to_datetime('2020/3/24')]
events_Ireland = [pd.to_datetime('2020/3/12')]
events_Israel = [pd.to_datetime('2020/3/15')]
events_Italy = [pd.to_datetime('2020/3/9')]
events_Japan = [pd.to_datetime('2020/4/7')]
events_Mexico = [pd.to_datetime('2020/3/23')]
events_Netherlands = [pd.to_datetime('2020/3/15')]
events_Pakistan = [pd.to_datetime('2020/3/24')]
events_Peru = [pd.to_datetime('2020/3/16')]
events_Poland = [pd.to_datetime('2020/3/24')]
events_Portugal = [pd.to_datetime('2020/3/19')]
events_Qatar = [pd.to_datetime('2020/3/23')]
events_Romania = [pd.to_datetime('2020/3/23')]
events_Russia = [pd.to_datetime('2020/3/30')]
events_Saudi_Arabia = [pd.to_datetime('2020/3/24')]
events_Singapore = [pd.to_datetime('2020/4/7')]
events_South_Korea = [pd.to_datetime('2020/2/25')]
events_Spain = [pd.to_datetime('2020/3/13')]
events_Sweden = [pd.to_datetime('2020/3/24')]
events_Switzerland = [pd.to_datetime('2020/3/16')]
events_Turkey = [pd.to_datetime('2020/3/21')]
events_United_Arab_Emirates = [pd.to_datetime('2020/3/31')]
events_Ukraine = [pd.to_datetime('2020/3/25')]
events_United_Kingdom = [pd.to_datetime('2020/3/24')]
events_United_States = [pd.to_datetime('2020/3/19')]

# City and province level
events_Tokyo = [pd.to_datetime('2020/4/7')]


events_all = {
    "Austria": events_Austria, "Belarus": events_Belarus, "Belgium": events_Belgium, "Brazil": events_Brazil,
    "Canada": events_Canada, "Chile": events_Chile, "China": events_China, "Ecuador": events_Ecuador,
    "France": events_France, "Germany": events_Germany, "India": events_India, "Indonesia": events_Indonesia, "Iran": events_Iran,
    "Ireland": events_Ireland, "Israel": events_Israel, "Italy": events_Italy, "Japan": events_Japan,
    "Mexico": events_Mexico, "Netherlands": events_Netherlands, "Pakistan": events_Pakistan, "Peru": events_Peru,
    "Poland": events_Poland, "Portugal": events_Portugal, "Qatar": events_Qatar, "Romania": events_Romania,
    "Russia": events_Austria, "Saudi Arabia": events_Saudi_Arabia, "Singapore": events_Singapore, "Korea, South": events_South_Korea,
    "Spain": events_Spain, "Sweden": events_Sweden, "Switzerland": events_Switzerland, "Turkey": events_Turkey,
    "United Arab Emirates": events_United_Arab_Emirates, "Ukraine": events_Ukraine, "United Kingdom": events_United_Kingdom,
    "US": events_United_States, "Tokyo": events_Tokyo
}

sigma_given = 0.4  # hyperparameter for LNML of Gaussian
# upper-bounds for 1st and 2nd D-MDL
alpha_1, alpha_2 = calculate_alpha({"sigma_given": sigma_given})

print(alpha_1)
print(alpha_2)

parameters = {"sigma_given": sigma_given,
              "alpha_1": alpha_1, "alpha_2": alpha_2}

os.makedirs(output_path, exist_ok=True)

alarms_for_each_country = pd.DataFrame()


# calculate metrics
country_stat = partial(_country_stat, events_all=events_all, params=parameters)
p = Pool(multi.cpu_count() - 1)
#args = list(range(1, n_samples))
args = countries
res = np.array(p.map(country_stat, args))

for i in range(len(res)):
    alarms_for_each_country = pd.concat(
        [alarms_for_each_country, res[i][17]], axis=0)
    country_graph(res[i][0], res[i][1], res[i][2], res[i][3], res[i][4], res[i][5],
                  res[i][6], res[i][7], res[i][8], res[
                      i][9], res[i][10], res[i][11],
                  res[i][12], res[i][13], res[i][14], res[i][15])

alarms_for_each_country.to_csv(
    'data/exponential_alarm_results.csv', index=False)
