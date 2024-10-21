from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

demonstrations_auth_endpoint = [
    pd.read_csv('demonstrations_auth_endpoint/2024-10-17T18-26-49.csv'),
    pd.read_csv('demonstrations_auth_endpoint/2024-10-18T11-24-13.csv'),
    pd.read_csv('demonstrations_auth_endpoint/2024-10-18T15-13-52.csv'),
]

demonstrations_search_endpoint = [
    pd.read_csv('demonstrations_search_endpoint/2024-10-12T18-19-36.csv'),
    pd.read_csv('demonstrations_search_endpoint/2024-10-13T18-29-29.csv'),
    pd.read_csv('demonstrations_search_endpoint/2024-10-13T22-55-07.csv'),
]

no_demonstrations_auth_endpoint = [
    pd.read_csv('no_demonstrations_auth_endpoint/2024-10-20T15-48-36.csv'),
    pd.read_csv('no_demonstrations_auth_endpoint/2024-10-20T23-35-21.csv'),
    pd.read_csv('no_demonstrations_auth_endpoint/2024-10-21T09-18-47.csv'),
]

no_demonstrations_search_endpoint = [
    pd.read_csv('no_demonstrations_search_endpoint/2024-10-10T19-16-18.csv'),
    pd.read_csv('no_demonstrations_search_endpoint/2024-10-11T12-03-56.csv'),
    pd.read_csv('no_demonstrations_search_endpoint/2024-10-12T01-46-09.csv'),
]

random_auth_endpoint = [
    pd.read_csv('random_auth_endpoint/2024-10-19T14-31-19.csv'),
    pd.read_csv('random_auth_endpoint/2024-10-19T19-28-30.csv'),
    pd.read_csv('random_auth_endpoint/2024-10-20T01-08-40.csv'),
]

random_search_endpoint = [
    pd.read_csv('random_search_endpoint/2024-10-14T09-21-35.csv'),
    pd.read_csv('random_search_endpoint/2024-10-14T17-39-33.csv'),
    pd.read_csv('random_search_endpoint/2024-10-14T23-13-27.csv'),
]


def show_plot(df_dict: Dict[str, pd.DataFrame]):
    df = pd.DataFrame(df_dict)

    plot = sns.relplot(
        data=df,
        kind='line',
    )

    plot.set_axis_labels(
        x_var='Episode',
        y_var='Mean Cumulative Reward',
    )
    
    plot.figure.suptitle('Mean Cumulative Reward over Episodes')

    plt.show()


search_endpoint_df_dict = {
    'With Demonstrations': pd.concat(demonstrations_search_endpoint)['Mean Cumulative Episodic Reward'],
    'Without Demonstrations': pd.concat(no_demonstrations_search_endpoint)['Mean Cumulative Episodic Reward'],
    'Random': pd.concat(random_search_endpoint)['Mean Cumulative Episodic Reward'],
}

auth_endpoint_df_dict = {
    'With Demonstrations': pd.concat(demonstrations_auth_endpoint)['Mean Cumulative Episodic Reward'],
    'Without Demonstrations': pd.concat(no_demonstrations_auth_endpoint)['Mean Cumulative Episodic Reward'],
    'Random': pd.concat(random_auth_endpoint)['Mean Cumulative Episodic Reward'],
}

show_plot(search_endpoint_df_dict)
show_plot(auth_endpoint_df_dict)