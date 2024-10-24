import os
from typing import Dict
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


def show_plot(
    demonstrations: pd.DataFrame,
    no_demonstrations: pd.DataFrame,
    random: pd.DataFrame,
    filename: str,
):
    df = pd.DataFrame({
        'With Demonstrations': pd.concat(demonstrations)['Mean Cumulative Episodic Reward'],
        'Without Demonstrations': pd.concat(no_demonstrations)['Mean Cumulative Episodic Reward'],
        'Random': pd.concat(random)['Mean Cumulative Episodic Reward'],
    })

    plot = sns.relplot(
        data=df,
        kind='line',
        palette=['#c91818', '#1742c7', '#585858',],
        linewidth=1.0,
        dashes={
            'With Demonstrations': '',
            'Without Demonstrations': '',
            'Random': '',
        },
    )

    plot.set_axis_labels(
        x_var='Episode',
        y_var='Mean Cumulative Reward',
    )
    
    plot.figure.set_figwidth(6.5)
    
    plot.figure.tight_layout()

    sns.move_legend(plot, 'lower center', ncol=3)
    
    plot.figure.subplots_adjust(top=0.9, bottom=0.18)
    
    plot.figure.suptitle('Mean Cumulative Reward over Episodes')

    file_path = os.path.dirname(__file__) + '/' + filename
    plot.figure.savefig(file_path, dpi=600)

show_plot(
    demonstrations=demonstrations_search_endpoint,
    no_demonstrations=no_demonstrations_search_endpoint,
    random=random_search_endpoint,
    filename='search_endpoint.png',
)

show_plot(
    demonstrations=demonstrations_auth_endpoint,
    no_demonstrations=no_demonstrations_auth_endpoint,
    random=random_auth_endpoint,
    filename='auth_endpoint.png',
)