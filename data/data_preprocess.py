import pandas as pd
import datetime as dt
from copy import deepcopy
import requests
import numpy as np
from datetime import timedelta


# Download raw data
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
filename = "time_series_covid19_confirmed_global.csv"
open(filename, 'wb').write(requests.get(url).content)
cases_origin = pd.read_csv(filename)

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
filename = "time_series_covid19_deaths_global.csv"
open(filename, 'wb').write(requests.get(url).content)
deaths_origin = pd.read_csv(filename)

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
filename = "time_series_covid19_recovered_global.csv"
open(filename, 'wb').write(requests.get(url).content)
recovered_origin = pd.read_csv(filename)


def preprocessing(country, cases_origin, deaths_origin, recovered_origin):
    cases = deepcopy(cases_origin)
    deaths = deepcopy(deaths_origin)
    recovered = deepcopy(recovered_origin)

    if country != "China" and country != "Canada":
        # Cumulative Deaths
        deaths = deaths[(deaths["Country/Region"] == country) &
                        (deaths["Province/State"].isnull() == True)].T.iloc[4:]
        # Cumulative Cases
        cases = cases[(cases["Country/Region"] == country) &
                      (cases["Province/State"].isnull() == True)].T.iloc[4:]
        # Cumulative Recovers
        recovered = recovered[(recovered["Country/Region"] == country)
                              & (recovered["Province/State"].isnull() == True)].T.iloc[4:]

    else:
        deaths = deaths[(deaths["Country/Region"] == country)
                        ].iloc[:, 4:].sum().T
        cases = cases[(cases["Country/Region"] == country)].iloc[:, 4:].sum().T
        recovered = recovered[
            (recovered["Country/Region"] == country)].iloc[:, 4:].sum().T

    deaths = deaths.reset_index()
    deaths.columns = ["date", "deaths"]
    deaths['date'] = pd.to_datetime(deaths['date'])

    cases = cases.reset_index()
    cases.columns = ["date", "cases"]
    cases["date"] = pd.to_datetime(cases["date"])

    recovered = recovered.reset_index()
    recovered.columns = ["date", "recovered"]
    recovered['date'] = pd.to_datetime(recovered['date'])

    cases["cases"] = cases["cases"].diff()
    cases = cases[1:]

    deaths["deaths"] = deaths["deaths"].diff()
    deaths = deaths[1:].reset_index(drop=True)

    recovered["recovered"] = recovered["recovered"].diff()
    recovered = recovered[1:].reset_index(drop=True)

    cases = pd.merge(cases, deaths, on="date", how="left")
    cases = pd.merge(cases, recovered, on="date", how="left")

    cases = cases.fillna(0)
    cases = cases.sort_values("date", ascending=False).reset_index(drop=True)

    cases["day"] = cases["date"].dt.day
    cases["month"] = cases["date"].dt.month
    cases["year"] = cases["date"].dt.year

    if country == "Korea, South":
        ctr = "South_Korea"
    elif country == "US":
        ctr = "United_States_of_America"
    elif country == "Saudi Arabia":
        ctr = "Saudi_Arabia"
    elif country == "United Arab Emirates":
        ctr = "United_Arab_Emirates"
    elif country == "United Kingdom":
        ctr = "United_Kingdom"
    else:
        ctr = country

    cases["countriesAndTerritories"] = ctr
    cases["date"] = cases["date"].dt.strftime('%d/%m/%Y')
    cases = cases.rename(columns={'date': 'dateRep'})

    cases = cases.reindex(columns=['dateRep', 'cases', 'deaths',
                                   'recovered', 'day', 'month', 'year', 'countriesAndTerritories'])

    return cases

# All countries
countries = ["Austria", "Belarus", "Belgium", "Brazil", "Canada", "Chile", "China", "Ecuador",
             "France", "Germany", "India", "Indonesia", "Iran",  "Ireland", "Israel", "Italy", "Japan", "Mexico",
             "Netherlands", "Pakistan", "Peru", "Poland", "Portugal", "Qatar", "Romania", "Russia",
             "Saudi Arabia", "Singapore", "Korea, South", "Spain", "Sweden", "Switzerland", "Turkey",
             "United Arab Emirates", "Ukraine", "United Kingdom", "US"]


df_all = pd.DataFrame()

for country in countries:
    ret = preprocessing(country, cases_origin=cases_origin,
                        deaths_origin=deaths_origin, recovered_origin=recovered_origin)
    df_all = pd.concat([df_all, ret], axis=0)

# tokyo data preparation
data_url_tokyo = 'https://stopcovid19.metro.tokyo.lg.jp/data/130001_tokyo_covid19_patients.csv'
filename_tokyo = 'tokyo.csv'
open(filename_tokyo, 'wb').write(requests.get(data_url_tokyo).content)
df_all_tokyo = pd.read_csv(filename_tokyo)
df_all_tokyo = df_all_tokyo[['公表_年月日']]
df_all_tokyo = df_all_tokyo.rename(columns={'公表_年月日': 'date'})
df_all_tokyo = df_all_tokyo.groupby('date').size().reset_index()
df_all_tokyo.columns = ['date', 'cases']
df_all_tokyo['date'] = pd.to_datetime(df_all_tokyo['date'])
start = df_all_tokyo['date'][0]
end = df_all_tokyo['date'][len(df_all_tokyo['date']) - 1]
df_tokyo = pd.DataFrame(
    {'date': np.arange(start, end + timedelta(days=1), timedelta(days=1))})
df_tokyo = pd.merge(df_tokyo, df_all_tokyo, on='date', how='left')
df_tokyo = df_tokyo.fillna(0)
df_tokyo["countriesAndTerritories"] = "Tokyo"
df_tokyo["deaths"] = 0
df_tokyo["recovered"] = 0
df_tokyo["day"] = df_tokyo["date"].dt.day
df_tokyo["month"] = df_tokyo["date"].dt.month
df_tokyo["year"] = df_tokyo["date"].dt.year
df_tokyo = df_tokyo.sort_values("date", ascending=False).reset_index(drop=True)
df_tokyo["date"] = df_tokyo["date"].dt.strftime('%d/%m/%Y')
df_tokyo = df_tokyo.rename(columns={'date': 'dateRep'})
df_tokyo = df_tokyo.reindex(columns=['dateRep', 'cases', 'deaths',
                                     'recovered', 'day', 'month', 'year', 'countriesAndTerritories'])

df_all = pd.concat([df_all, df_tokyo], axis=0)

#df_all.reindex(columns=['dateRep', 'cases' ,'deaths', 'recovered', 'day', 'month', 'year', 'countriesAndTerritories'])

df_all.to_csv("covid19.csv", index=False, encoding="utf-8")
