# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
from sklearn.linear_model import Lasso
import pickle

app = dash.Dash(__name__)

colors = {
    'background': 'white',
    'text': 'black'
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv('NFLX.csv')

filename = "lasso_regression.sav"
model = pickle.load(open(filename, 'rb'))

fig2 = px.line(df['Close'],
               labels={
                     "index": "Index",
                     "value": "Close Price ($)",
                 })

'''
fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)
'''

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='STOCK ANALYSIS',
        style={
            'textAlign': 'center',
            'color': colors['text'],

        }
    ),

    html.Div(children='Analysing and Predicting the Stock Market', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='GRAPH OF CLOSE VALUE OVER TIME USING ARIMA MODEL',
        figure=fig2
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)