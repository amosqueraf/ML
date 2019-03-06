import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly.plotly as py

datos = {
    '+10':[1,2,3,4],
    '+20':[2,3,6,12],
    '-10':[4,3,2,1],
    '-20':[12,6,3,2]}

app = dash.Dash()

markdown_text = '''
## IIND 4101 - Herramientas Computacionales para el Analisis de Datos
### Aplicaciones Web Interactivas: poder analitico de Python + visualizacion versatil y "responsiva"

Ya sabemos usar Markdown, y resulta muy sencillo hacer aplicaciones que incorporen, por ejemplo:

* Texto
* Imagen
* Ecuaciones

En este caso, a traves de Dash, podemos incorporar elementos de control que interactuan con visualizaciones que responden a comandos del usuario.
'''

'''
    LAYOUT
'''
app.layout = html.Div([ html.Div([
                          dcc.Markdown(children=markdown_text)
                            ]),

                        html.Div([
                            html.H3('Evaluar escenarios:', style={'textAlign': 'left'}),
                            dcc.RadioItems(
                                    options = [
                                        {'label': 'Reduccion 10%', 'value': '-10'},
                                        {'label': 'Reduccion 20%', 'value': '-20'},
                                        {'label': 'Aumento 10%', 'value': '+10'},
                                        {'label': 'Aumento 20%', 'value': '+20'} ],
                                    value = '+10',
                                    id = 'boton')
                            ]),                        

                        html.Div([     
                            dcc.Graph(
                                id='figura',
                                figure = {    'data': [ {'x': [1,2,3,4], 'y': datos['+10'], 'type': 'bar'} ],
                                            'layout': { 'title': 'Demanda insatisfecha por zona: escenario "+10"',
                                                    'height':300}
                                        }
                            )]),
                                                
                        html.Div( [ html.H5('Universidad de los Andes - Todos los derechos reservados', style={'textAlign': 'center'}) ] )
                        ], style = {'width': '80%', 'align':'center'})


@app.callback(
    dash.dependencies.Output(component_id='figura', component_property='figure'),
    [dash.dependencies.Input(component_id='boton', component_property='value')])
def update_output(escenario):

    fig = { 'data': [ {'x': [1,2,3,4], 'y': datos[escenario], 'type': 'bar'} ],
            'layout': { 'title': 'Demanda insatisfecha por zona: escenario "'+escenario+'"',
                        'height': 400}
                                        }
    return fig

if __name__ == '__main__':
    app.run_server(debug = True)