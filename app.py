# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

colors = {
    'background': 'white',
    'text': 'black'
}

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_csv('NFLX.csv')

filename = "ridge_regression.sav"

model = pickle.load(open(filename, 'rb'))

candlestick = go.Figure(data=[go.Candlestick(x=df['Date'],
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])

candlestick.update_layout(title='CANDLESTICK GRAPH', title_x=0.5)

line_graph = px.scatter(x=df['Open'], y=df['Close'], labels={'x': 'Open', 'y': 'Close'})
line_graph.update_layout(title='OPEN v CLOSE', title_x=0.5)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='STOCK ANALYSIS',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-weight': 'bold'
        }
    ),

    html.Div(children='Analysing and Predicting the Stock Market', style={
        'textAlign': 'center',
        'color': colors['text'],
        'fontSize': "18px",
        "fontWeight": "bold"
    }),

    dcc.Graph(
        id='CANDLESTICK',
        figure=candlestick
    ),

    html.H1(
        children='PREDICTING CLOSE VALUE',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'font-weight': 'bold'
        }
    ),

    html.Br(),

    html.Div([
        html.Span(
            children='Input Open Price ($): ',
            style={
                'color': colors['text'],
                "font-size": "20px",
                'margin-left': '230px'
            }
        ),
        dcc.Input(
            id="input_value",
            value="400",
            type="text",
            style={
                "font-size": "18px"
            }
        ),
        html.Span(
            id="predicted_close",
            style={
                "font-size": "20px",
                "margin-left":"90px",
            }
        )
    ]),

    html.Br(),
    html.Br(),

    dcc.Graph(
        id='CLOSE-OPEN',
        figure=line_graph
    )

])


@app.callback(
    Output(component_id="predicted_close", component_property="children"),
    Input(component_id="input_value", component_property="value")
)
def update_output_div(input_value):
    input_value = np.array(float(input_value))
    input_value = input_value.reshape(-1, 1)
    close_price = model.predict(input_value)
    return 'Predicted Output: {}, Accuracy: 92.16%'.format(close_price)


if __name__ == '__main__':
    app.run_server(debug=True)
