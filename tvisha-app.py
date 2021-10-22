# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc, callback_context
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

filename = "linear_regression_aapl.pickle"

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
            html.H4("Stock Analytic Dashboard", className="display-3", style={"font-size": "20px"}),
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
            {'label': 'GOOGLE', 'value': 'GOOG-2.csv'}
        ],
        value='NFLX.csv'
    ),
    dcc.Dropdown(
        id='date',
        options=[
            {'label': '1mo', 'value': '25'},
            {'label': '6mo', 'value': '50'},
            {'label': '1y', 'value': '300'},
            {'label': '5y', 'value': 'default'}
        ],
        value='default'
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
    html.Div(
        id='button-output-container',
        children=html.Div(dcc.Graph(
            id='CLOSE-OPEN',
            figure=line_graph,
        ),
        )),
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
                            "View Graph", color="primary", className="mt-auto", id="AM-button", n_clicks=0, name="AMZN.csv"
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
                            "View Graph", color="primary", className="mt-auto", id="AP-button", n_clicks=0, name="AAPL.csv"
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
                            "View Graph", color="primary", className="mt-auto", id="FB-button", n_clicks=0, name="FB.csv"
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
                            "View Graph", color="primary", className="mt-auto", id="GOOG-button", n_clicks=0, name="GOOG-2.csv"
                        ),
                    ]
                )
            ),
            dbc.Card(
                dbc.CardBody(
                    [

                        html.H5("Netflix", className="card-title"),
                        html.P(
                            "To check stocks from over 5 years and predict to make better investments",

                            className="card-text",
                        ),
                        dbc.Button(
                            "View Graph", color="primary", className="mt-auto", id="NF-button", n_clicks=0, name="NFLX.csv"
                        ),
                    ]
                )
            ),
        ]
    )

])


# Open and Close Predictor
@app.callback(
    Output(component_id="predicted_close", component_property="children"),
    Input(component_id="input_value", component_property="value")

)
def update_output_div(input_value):
    input_value = np.array(float(input_value))
    input_value = input_value.reshape(-1, 1)
    close_price = model.predict(input_value)
    return 'Predicted Close Price ($): {}'.format(close_price)


# Scatter plot
@app.callback(
    Output('button-output-container', 'children'),
    Input('AM-button', 'n_clicks'),
    Input('AP-button', 'n_clicks'),
    Input('FB-button', 'n_clicks'),
    Input('GOOG-button', 'n_clicks'),
    Input('NF-button', 'n_clicks')
)
def update_output(btn1,btn2,btn3,btn4,btn5):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'AM-button' in changed_id:
        msg = 'AMZN.csv'
    elif 'AP-button' in changed_id:
        msg = 'AAPL.csv'
    elif 'FB-button' in changed_id:
        msg = 'FB.csv'
    elif 'GOOG-button' in changed_id:
        msg = 'GOOG-2.csv'
    elif 'NF-button' in changed_id:
        msg = 'NFLX.csv'
    else:
        msg = 'NFLX.csv'
    df = pd.read_csv(msg)
    line_graph = px.scatter(x=df['Open'], y=df['Close'], labels={'x': 'Open', 'y': 'Close'})
    line_graph.update_layout(title='OPEN v CLOSE', title_x=0.5)
    return html.Div(dcc.Graph(
        id='CLOSE-OPEN',
        figure=line_graph
    ))


# Dropdown companies
@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value'),
    Input('date','value')
)
def update_output(value,date):
    # df = pd.read_csv(value)
    df = pd.read_csv(value)
    if date == 'default':
        df = df[:]
    else:
        start_date = len(df) - int(date)
        df = df[start_date:]
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


if __name__ == '__main__':
    app.run_server(debug=True)
