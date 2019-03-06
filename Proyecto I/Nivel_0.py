import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly.plotly as py

app = dash.Dash()

sup_izq = html.Div([     
                        dcc.Graph(
                                id = 'fig1',
                                figure = {  'data': [ {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar'} ],
                                        'layout': {'title': 'Sup Izq', 'height':300}    } 
                                )],
                    style={'width': '45%', 'display': 'inline-block'})

sup_der = html.Div([     
                        dcc.Graph(
                                id='fig2',
                                figure={    'data': [ {'x': [3, 2, 1], 'y': [4, 1, 2], 'type': 'scatter'} ],
                                            'layout': {'title': 'Sup Der', 'height':300}    }
                                )],
                    style={'width': '45%', 'display': 'inline-block', 'float': 'right'},)

inf_izq = html.Div([     
                        dcc.Graph(
                                id = 'fig3',
                                figure = {  'data': [ {'x': [1, 2, 3], 'y': [2, 1, 2], 'type': 'scatter'} ],
                                        'layout': {'title': 'Inf Izq', 'height':300}    } 
                                )],
                    style={'width': '45%', 'display': 'inline-block'})

inf_der = html.Div([     
                        dcc.Graph(
                                id='fig4',
                                figure={    'data': [ {'x': [3, 2, 1], 'y': [4, 1, 2], 'type': 'bar'} ],
                                            'layout': {'title': 'Inf Der', 'height':300}    }
                                )],
                    style={'width': '45%', 'display': 'inline-block', 'float': 'right'},)


bloq_sup = html.Div([   sup_izq,                            
                        sup_der
                        ])

bloq_inf = html.Div([   inf_der,                            
                        inf_izq
                        ])

banner = html.Div([
                    html.H2('IIND 4101 - Herramientas Computacionales para Analisis de Datos', style={'textAlign': 'center'}), 
                    html.H3('Creacion de aplicaciones web interactivas desde Python con Dash', style={'textAlign': 'center'})
                ])

footer = html.Div([  html.H4('Universidad de los Andes - Todos los derechos reservados', style={'textAlign': 'center'})  ])

'''
    LAYOUT
'''
app.layout = html.Div([ banner,
                        bloq_sup,                            
                        bloq_inf,
                        footer
                        ])


if __name__ == '__main__':
    app.run_server(debug=True)