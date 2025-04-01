#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import dash 
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd 
import plotly.express as px
import numpy as np 


# In[2]:


dataset = pd.read_csv("FIFA_data.csv", delimiter=';')


# In[16]:


app = dash.Dash()
server = app.server

countries = np.concatenate([np.array(dataset['Winners']), np.array(dataset['Runners-up'])])
countries = np.unique(countries)

world = px.data.gapminder()[['country']].drop_duplicates()

world['Won'] = world['country'].apply(lambda c: 'Has Won' if c in np.array(dataset['Winners'].unique()) else 'Not Won')

years = np.array(dataset.loc[::, 'Year'])

figure = px.choropleth(
    world,
    locations="country",
    locationmode="country names",
    color="Won",
    scope="world"
)


app.layout = html.Div([
    html.H1("FIFA World Cup Dashboard", style={'textAlign':'center'}),
    html.Div([
        dcc.Graph(figure=figure)
    ]),
    html.Div([
        html.Label("Select a Country"),
        dcc.Dropdown(countries, countries[0], id='country_dropdown'),

        html.Label("Select a Year"),
        dcc.Dropdown(years, years[0], id='year_dropdown')
    ]),

    html.Div([
        html.P(id="won_world_cup", style={'fontFamily':'Courier New, monospace', 'fontSize':'30px'}),
        html.P(id="year_winner_runnerup", style={'fontFamily': 'Courier New, monospace','fontSize':'30px'})
    ])
])


@callback(
    [Output('won_world_cup', 'children'),
     Output('year_winner_runnerup', 'children')],
    [Input('country_dropdown', 'value'),
     Input('year_dropdown', 'value')]
)

def get_world_cup_info(country, year):
    dataset = pd.read_csv('FIFA_data.csv', delimiter=';')
    winner_frequency = dataset['Winners'].value_counts()

    if country in winner_frequency.keys():
        frequency = winner_frequency[country]
    else:
        frequency = 0

    row = dataset.loc[dataset.loc[::, 'Year'] == year]
    index = row['ID']
    winner = np.array(row['Winners'][index])
    runnerup = np.array(row['Runners-up'][index])

    country_won = f"{country} won {frequency} times."
    winner_runnerup = f"{year} Winner: {winner[0]}, Runner-up: {runnerup[0]}"

    return country_won, winner_runnerup
    

if __name__ == "__main__":
    app.run(debug=False)



# In[ ]:




