import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly.plotly as py

app = dash.Dash()

markdown_text = '''
## IIND 4101 - Herramientas Computacionales para el Analisis de Datos
### Aplicaciones Web Interactivas: poder analitico de Python + visualizacion versatil y responsiva

Ya sabemos usar Markdown, y resulta muy sencillo hacer aplicaciones que incorporen, por ejemplo:

* texto
* imagen
* ecuaciones

En este caso, a traves de Dash, podemos incorporar elementos de control que interactuan con visualizaciones que responden a comandos del usuario.
'''

'''
    LAYOUT
'''
app.layout = html.Div([ html.Div([
                          dcc.Markdown(children=markdown_text)
                            ]),

                        html.Div([
                            html.H4('Evaluar escenarios:', style={'textAlign': 'left'}),
                            dcc.RadioItems(
                                    options = [
                                        {'label': 'Reduccion 10%', 'value': '-10'},
                                        {'label': 'Reduccion 20%', 'value': '-20'},
                                        {'label': 'Aumento 10%', 'value': '10'},
                                        {'label': 'Aumento 20%', 'value': '20'} ],
                                    value = '10')
                            ]),                        

                        html.Div([     
                            dcc.Graph(
                                id='fig4',
                                figure={    'data': [ {'x': [3, 2, 1], 'y': [4, 1, 2], 'type': 'bar'} ],
                                            'layout': { 'title': 'Demanda insatisfecha por zona: escenario tal',
                                                    'height':300}
                                        }
                            )]),
                                                
                        html.Div( [ html.H5('Universidad de los Andes - Todos los derechos reservados', style={'textAlign': 'center'}) ] )
                        ])

if __name__ == '__main__':
    app.run_server(debug = True)

'''

'''