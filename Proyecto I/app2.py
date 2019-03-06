# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:58:38 2018

@author: dretrepo
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import accuracy_score


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


viajes = pd.read_pickle("viajes3.pkl")
viajes2=viajes
viajes3=viajes
viajes3 = viajes3.drop(['VICTIMA_AGRESION'],1)
probab = 0.15

"""
viajes2 = pd.get_dummies(viajes2, columns=['MEDIO_PREDOMINANTE','ZAT_DESTINO','ZAT_ORIGEN','SEXO','ESTRATO_y'])
X = viajes2.drop(['VICTIMA_AGRESION'],1)
y = viajes2['VICTIMA_AGRESION']

model = linear_model.LogisticRegression()
model.fit(X,y)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)
name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
"""
app.layout = html.Div(children=[
html.Div([
    html.Div([##bloqueA
        html.H1(children='BOGOTÁ MOVILITY RISK',style={'margin':'auto','width': "50%",'color': 'brown', 'fontSize': 40}),
        html.Div(children='''
            Aplicación para el cálculo de la probabilidad de ser agredido al movilizarse en Bogotá
        ''',style={'margin':'auto','width': "70%",'color': 'brown', 'fontSize': 20}),
    ]),
],style={'width': '100%', 'display': 'inline-block','color': 'brown', 'fontSize': 40}),
     
  html.Div([##bloqueA
          html.Div([##cuad1
        
                html.Label('Digite su edad:'),
                dcc.Input( id='Edad',
                min=viajes['EDAD'].min(),
                max=viajes['EDAD'].max(),
                value=viajes['EDAD'].max(), type='integer'),
                
                html.Label('Seleccione su Género'),
                dcc.Dropdown(id='Genero',
                    options=[
                        {'label': 'Masculino', 'value': 'Hombre'},
                        {'label': 'Femenino', 'value': 'Mujer'}
                    ],
                   value='Hombre',
                   style={'width': '50%'}
                   ),
                
                html.Label('Seleccione su Estrato'),
                dcc.RadioItems(id='Estrato',
                        options=[{'label': i, 'value': i} for i in sorted(viajes['ESTRATO_y'].unique())],
                        value=3,
                        labelStyle={'display': 'inline-block'}
                        
                    ),
                
                html.Label('Digite el tiempo promedio del viaje (minutos):'),
                dcc.Input( id='minutos',
                min=viajes['DURACION_MIN'].min(),
                max=viajes['DURACION_MIN'].max(),
                value=50, type='integer'),
                          
                html.Label('Seleccione el número de etapas:'),
                dcc.Dropdown(id='etapas',
                        options=[{'label': i, 'value': i} for i in sorted(viajes['CANTIDAD_ETAPAS'].unique())],
                    value=3
                ,style={'margin':'auto','float':'left','width': '40%', 'display': 'inline-block'}
                ),
                    
                 html.Button('Submit', id='button'),
                 html.Div(id='output-container-button', children='Enter a value and press submit')
            ],style={'margin':'auto','width': '50%', 'display': 'inline-block'}
            ), ##cierre cuad1
                
            html.Div([ ##cuad2       
                    html.Label('Seleccione su medio de transporte principal:'),
                    dcc.Dropdown(id='transporte',
                        options=[{'label': i, 'value': i} for i in viajes['MEDIO_PREDOMINANTE'].unique()],
                        value='Transmilenio',
                        style={'width': '50%'}
                    ),
                    
                    html.Label('Seleccione el dia a realizar el viaje:'),
                    dcc.Dropdown(id='dia',
                        options=[
                            {'label': 'Día laboral', 'value': '1'},
                            {'label': 'Fin de semana', 'value': '0'}
                        ],
                        value='1',
                        style={'width': '45%'}
                    ),
    
                    html.Label('Seleccione origen:'),
                    dcc.Dropdown(id='origen',
                            options=[{'label': i, 'value': i} for i in viajes['ZAT_ORIGEN'].unique()],
                        value=566,
                        style={'width': '60%'}
                    ),
                            
                    html.Label('Seleccione destino:'),
                    dcc.Dropdown(id='destino',
                            options=[{'label': i, 'value': i} for i in viajes['ZAT_DESTINO'].unique()],
                        value=238,
                        style={'width': '60%'}
                    ),

                    html.Label('Seleccione la hora de viaje'),          
                        dcc.Slider(id='hora',
                        min=viajes['HORA_DEC'].min(),
                        max=viajes['HORA_DEC'].max(),
                        marks={str(hora): str(hora) for hora in viajes['HORA_DEC'].unique()},
                        value=viajes['HORA_DEC'].max()
                    ),
            ],style={'width': '50%', 'display': 'inline-block'}
            )##cierre cuad2
    ]),##cierre bloque A

  html.Div([##bloqueB
          html.Div([##cuad3
                    html.Div([ ##cuad4.1
                        dcc.Graph(
    						id='bar_plot',      
    								figure= go.Figure(data=[go.Bar(x=['Probabilidad'],y=[probab],name='Probabilidad'), 
    														go.Bar(x=['Probabilidad'],y=[0.5-(probab*0.5)],name='')],
    								   layout=go.Layout(barmode='stack', title='Probabilidad de agresión'),
    								   )
    								),
                        html.Div([
                            html.Label('Según los parametros ingresados la probabilidad de ser agredido es de (%):'),   
                            html.Div(probab*100, style={'color': 'blue', 'fontSize': 40}),
                          
                        ], 
                        style={'marginBottom': 50, 'marginTop': 25}),
 ##                      html.H1(children=probab*100),
                    ],style={'width': '30%', 'display': 'inline-block','color': 'blue', 'fontSize': 18}
                    ), ##cierre cuad4.1

                   html.Div([ ##cuad4.2
                         dcc.Graph(
                            id='HORA-vs-AGRE',
                            figure={
                                'data': [
                                    go.Scatter(
                                        x=viajes[viajes['VICTIMA_AGRESION'] == i]['HORA_DEC'].unique(),
                                        y=viajes[viajes['VICTIMA_AGRESION'] == i]['DURACION_MIN'],
                                        mode='markers',
                                        opacity=0.7,
                                        marker={
                                            'size': 15,
                                            'line': {'width': 0.5, 'color': 'white'}
                                        },
                                        name=i
                                        ) for i in viajes.VICTIMA_AGRESION.unique()
                                        ],
                                        'layout': go.Layout(
                                            xaxis=dict(dtick=1,title='HORA DE INICIO DEL VIAJE (HH)',ticklen=viajes['HORA_DEC'].max()),
                                            yaxis={'title': 'DURACION DEL VIAJE (MIN)'},
                                            hovermode='closest', title='Agresiones según hora de viaje y su duración'
                                        )
                                    }
                                ),
                        ],style={'width': '70%', 'display': 'inline-block'}
                                
                    ), ##cierre cuad4.2 """

                ],style={'width': '100%', 'display': 'inline-block'}
            ), ##cierre cuad3
                
            html.Div([ ##cuad4
                       dcc.Graph(
    						id='bar_plot2',      
    								figure= go.Figure(data=[go.Bar(x=['Probabilidad'],y=[probab],name='Probabilidad'), 
    														go.Bar(x=['Probabilidad'],y=[1-probab],name='')],
    								   layout=go.Layout(barmode='stack', title='Aqui va el mapa'),
    								   )
    								),
            ],style={'width': '85%', 'float': 'right', 'display': 'inline-block','backgroundColor': 'pink'}
            )##cierre cuad4
    ]),##cierre bloque B                             
],style={'backgroundColor': 'white'})
                                
@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('Edad', 'value'),
     dash.dependencies.State('Genero', 'value'),
     dash.dependencies.State('Estrato', 'value'),
     dash.dependencies.State('transporte', 'value'),
     dash.dependencies.State('dia', 'value'),
     dash.dependencies.State('origen', 'value'),
     dash.dependencies.State('destino', 'value'),
     dash.dependencies.State('hora', 'value'),
     dash.dependencies.State('etapas', 'value'),
     dash.dependencies.State('minutos', 'value')
    ])
def update_output(n_clicks, Edad,Genero,Estrato,transporte,dia,origen,destino,hora,etapas,minutos):
    
    ex_dic = {
            'MEDIO_PREDOMINANTE':transporte,
            'ZAT_DESTINO':destino,
            'ZAT_ORIGEN':origen,
            'HABIL':dia,
            'DISTANCIA': viajes['DISTANCIA'][(viajes['ZAT_ORIGEN']==origen)&(viajes['ZAT_DESTINO']==destino)].mean(),
            'CANTIDAD_ETAPAS': etapas,
            'DURACION_MIN': minutos,
            'SEXO': Genero,
            'EDAD': Edad,
            'ESTRATO_y': Estrato,
            'HORA_DEC': hora } 

    input_df = pd.DataFrame( ex_dic, index=[150000])
    viajes4=viajes3.append(input_df,ignore_index=False)
    input_df_dummy = pd.get_dummies(viajes4, columns=['MEDIO_PREDOMINANTE','ZAT_DESTINO','ZAT_ORIGEN','SEXO','ESTRATO_y'])
    input_df2 = pd.DataFrame( input_df_dummy.loc[[150000]])
    probab = pd.DataFrame(model.predict_proba(input_df2))[1]    
  
    print(probab)
    
    return 'Edad: "{}", Genero: "{}", Estrato: "{}", transporte: "{}", dia: "{}", origen: "{}", destino: "{}", hora: "{}", n_clicks: "{}" '.format(
        Edad,Genero,Estrato,transporte,dia,origen,destino,hora, n_clicks     
)

if __name__ == '__main__':
    app.run_server(debug=True)