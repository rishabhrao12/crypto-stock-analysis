# Run this app with `tvisha python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# Tvisha change the html and css elements to change the design


import dash
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import html

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

colors = {
    'background': 'white',
    'text': 'black'
}

filename = "ridge_regression.sav"

model = pickle.load(open(filename, 'rb'))

df = pd.read_csv('NFLX.csv')
candlestick = go.Figure(data=[go.Candlestick(x=df['Date'],
                                             open=df['Open'],
                                             high=df['High'],
                                             low=df['Low'],
                                             close=df['Close'])])

candlestick.update_layout(title='CANDLESTICK GRAPH', title_x=0.5)

line_graph = px.scatter(x=df['Open'], y=df['Close'], labels={'x': 'Open', 'y': 'Close'})
line_graph.update_layout(title='OPEN v CLOSE', title_x=0.5)

'''arima_model = pickle.load(open("ARIMA_Model.pkl", 'rb'))
fig = arima_model.plot_predict(1,60)'''

app.layout = html.Div([

    dbc.Jumbotron(
        [
            html.H1("Stock Analysis", className="display-3"),
            html.P(
                "Using machine learning to analyse "
                "and predict stock values",
                className="lead",
            ),
            html.Hr(className="my-2"),
            html.P(dbc.Button("Predict Now!", color="primary"), className="lead"),
            html.H4("Stock Analytics", className="display-3", style={"font-size": "20px"}),
        ],
    ),
    html.Hr(),
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'NETFLIX', 'value': 'NFLX.csv'},
            {'label': 'AMAZON', 'value': 'AMZN.csv'},
            {'label': 'APPLE', 'value': 'AAPL.csv'},
            {'label': 'FACEBOOK', 'value': 'FB.csv'},
            {'label': 'GOOGLE', 'value': 'GOOG.csv'}
        ],
        value='NFLX.csv'
    ),
    html.Div(
        id='dd-output-container',
        children=html.Div(dcc.Graph(
                id='CANDLESTICK',
                figure=candlestick
    ))
    ),

    # html.Hr(),

    html.Div(children='Predicting Close Value ', style={
        'textAlign': 'center',
        'color': 'black',
        'font-size': "24px",
        'font-family': 'Trebuchet MS',
        'background-color': '#F5F5F5',
        'margin': 'auto',
        'text-indent': '0px',
        'height': '70px',

    }),
    html.Div(style={'background-color': '#F5F5F5'}, children=[
        html.Span(
            children='Input Open Price ($): ',
            style={
                'margin-left': '150px',
                'color': 'black',
                "font-size": "20px",
                'font-family': 'Trebuchet MS',

            }
        ),
        dcc.Input(
            id="input_value",
            value="400",
            type="text",
            style={
                "font-size": "18px",
                'font-family': 'Trebuchet MS',
            }
        ),
        html.Span(
            id="predicted_close",
            style={
                "margin-left": '180px',
                "font-size": "20px",
                'font-family': "Trebuchet MS",
                'color': 'black',
            },
        ),

        #
        # html.Br(),
        # html.Br(),
        # html.Br(),
        html.Br(),

        html.Span(
            children="Model Accuracy: 99.16%",
            id="accuracy",
            style={
                "font-size": "20px",
                'font-family': 'Trebuchet MS',
                'color': 'black',
                'margin-left': '520px',

            },
        ),
        html.Div(
            [
                dbc.Progress(value=90, color="black", className="mb-3", style={"width": "2000px"}),

            ]
        )
    ]),

    html.Br(),
    html.Br(),

    dcc.Graph(
        id='CLOSE-OPEN',
        figure=line_graph,
    ),
    html.Hr(style={"color": "blue", "size": "10px"}),
    dbc.CardDeck(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Amazon", className="card-title"),
                        html.P(
                            "To check stocks from over 5 years and predict to make better investments  ",

                            className="card-text",
                        ),
                        dbc.Button(
                            "Click here", color="primary", className="mt-auto"
                        ),
                    ]
                )
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Apple", className="card-title"),
                        html.P(
                            "To check stocks from over 5 years and predict to make better investments ",
                            className="card-text",
                        ),
                        dbc.Button(
                            "Click here", color="primary", className="mt-auto"
                        ),
                    ]
                )
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Facebook", className="card-title"),
                        html.P(
                            "To check stocks from over 5 years and predict to make better investments ",

                            className="card-text",
                        ),
                        dbc.Button(
                            "Click here", color="primary", className="mt-auto"
                        ),
                    ]
                )
            ),
            dbc.Card(
                dbc.CardBody(
                    [

                        html.H5("Google", className="card-title"),
                        html.P(
                            "To check stocks from over 5 years and predict to make better investments ",

                            className="card-text",
                        ),
                        dbc.Button(
                            "Click here", color="primary", className="mt-auto"
                        ),
                    ]
                )
            ),
        ]
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
    return 'Predicted Close Price ($): {}'.format(close_price)


@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value')
)
def update_output(value):
    # df = pd.read_csv(value)
    df = pd.read_csv(value)
    candlestick = go.Figure(data=[go.Candlestick(x=df['Date'],
                                                 open=df['Open'],
                                                 high=df['High'],
                                                 low=df['Low'],
                                                 close=df['Close'])])

    candlestick.update_layout(title='CANDLESTICK GRAPH', title_x=0.5)

    return html.Div(dcc.Graph(
        id='CANDLESTICK',
        figure=candlestick
    ))

    #line_graph = px.scatter(x=df['Open'], y=df['Close'], labels={'x': 'Open', 'y': 'Close'})
    #line_graph.update_layout(title='OPEN v CLOSE', title_x=0.5)


if __name__ == '__main__':
    app.run_server(debug=True)
