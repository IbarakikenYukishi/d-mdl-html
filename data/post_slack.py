import requests
import json
import pandas as pd
import sys
webhook_url = "https://hooks.slack.com/services/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"

args = sys.argv
# print(args)
channel = '#' + args[1]
if channel not in ["#sandbox", "#daily_updates"]:
    raise ValueError("arg[1] is " + str(args[1]) +
                     ", but should be sandbox or daily_updates.")

countries = ["Japan", "Tokyo"]

if channel=="#daily_updates":
    requests.post(webhook_url, data=json.dumps({'text': 'Finished update.',
                                            'channel': channel}))
elif channel=="#sandbox":
    requests.post(webhook_url, data=json.dumps({'text': 'Finished push.',
                                            'channel': channel}))


def post_slack_for_each_country(df, countries):
    for country in countries:

        post_str = '*' + country + '*\n'
        post_str += '> Date: ' + \
            str(df[df["CountryAndTerritory"]
                   == country]["data_latest_date"].values[0]) + '\n'
        post_str += '> Daily confirmed cases: ' + \
            str(int(df[df["CountryAndTerritory"]
                       == country]["cases"].values)) + '\n'

        detected = ''
        if df[df["CountryAndTerritory"] == country]["0th"].values == 1:
            detected += '0th '
        if df[df["CountryAndTerritory"] == country]["1st"].values == 1:
            detected += '1st '
        if df[df["CountryAndTerritory"] == country]["2nd"].values == 1:
            detected += '2nd'

        if detected == '':
            post_str += '> Detected alarms: no alarms' + '\n'
        else:
            post_str += '> `Detected alarms:' + detected + '`\n'

        requests.post(webhook_url, data=json.dumps({'text': post_str,
                                                    'channel': channel}))

# Gaussian Result
df_gaussian = pd.read_csv("data/gaussian_alarm_results.csv")
requests.post(webhook_url, data=json.dumps({'text': '*------Gaussian Modeling------*',
                                            'channel': channel}))
post_slack_for_each_country(df_gaussian, countries)

# Gaussian Result
df_exponential = pd.read_csv("data/exponential_alarm_results.csv")
requests.post(webhook_url, data=json.dumps({'text': '*------Exponential Modeling------*',
                                            'channel': channel}))
post_slack_for_each_country(df_exponential, countries)
