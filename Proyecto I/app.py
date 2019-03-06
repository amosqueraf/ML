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

from arcgis.geocoding import geocode
import mapbox as mp
import plotly.graph_objs as go

import pickle
from pyproj import Geod


                                              # iniciar la app
wgs84_geod = Geod(ellps='WGS84')
def Distance(lat1,lon1,lat2,lon2):
  az12,az21,dist = wgs84_geod.inv(lon1,lat1,lon2,lat2) #Yes, this order is correct
  return dist

def closest_dest(data,lat,lon):
    data['distancia']=data.apply(lambda x: Distance(x['Latitud_Destino'],x['Longitud_Destino'],lat,lon), axis=1)   
    return data.loc[data['distancia'].idxmin()]['ZAT_DESTINO']

def closest_orig(data,lat,lon):
    data['distancia']=data.apply(lambda x: Distance(x['Latitud_Origen'],x['Longitud_Origen'],lat,lon), axis=1)   
    return data.loc[data['distancia'].idxmin()]['ZAT_ORIGEN']

    
viajes = pd.read_pickle("viajes3.pkl")
df_zats = pd.read_pickle("df_zats.pkl")
viajes2=viajes
viajes3=viajes
viajes3 = viajes3.drop(['VICTIMA_AGRESION'],1)
probab = 0.045


viajes2 = pd.get_dummies(viajes2, columns=['MEDIO_PREDOMINANTE','ZAT_DESTINO','ZAT_ORIGEN','SEXO','ESTRATO_y'])
X = viajes2.drop(['VICTIMA_AGRESION'],1)
y = viajes2['VICTIMA_AGRESION']
'''
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
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 '''
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))


gis = GIS() ### Hace parte de ArcGIS, dejar así
mapbox_access_token = 'pk.eyJ1IjoiZWRjaGFpbmNhIiwiYSI6ImNqcDBmMXB2dDAwOTQzd21ycW80cWE4NmQifQ.KsW-cUYzM-GljOxeQoT_Eg' ###Para obtener el token registrarse en mapbox
serviceRoutes = mp.Directions(access_token=mapbox_access_token)
rJson = {}
lat_origen1=4.667038470165603
long_origen1=-74.13272187468982
lat_destino1=4.722337466558932
long_destino1=-74.07237205536961
zat_destino=132
zat_orig=954
#########################Mapa de inicio de Bogotá###############################
dataIni=[go.Scattermapbox(
    showlegend = False,
    lat=[4.61496],
    lon=[-74.06941],
    )]
mapLayoutIni = go.Layout(
    autosize=True,
    height = 400,
    hovermode='closest',
    title='Mapa con Ruta',
    font=dict(color='#000101'),
    titlefont=dict(color='#000101', size=14),
    margin=dict(l=35,r=35,b=30,t=30),
    paper_bgcolor="#57A1E6", ###Con este cambian el fondo, recomiendo esta web https://www.w3schools.com/colors/colors_hexadecimal.asp
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="basic", #Estilo del mapa, puede ser "basic" también
        center=dict(
            lat=4.61496,
            lon=-74.06941
        ),
        pitch=0,
        zoom=9 #Acercamiento al mapa
    ))
figureIni = dict(data = dataIni,layout=mapLayoutIni)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True # Sirve para evitar errores al inicio de callbacks 
                                                # que dependan de elementos no creados al momento de 

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
            dcc.Graph(
                            id='HORA-vs-AGRE',
                            figure={
                                'data': [
                                    go.Scatter(
                                        x=viajes['HORA_DEC'][viajes['VICTIMA_AGRESION'] == '1'][viajes['DURACION_MIN']<300],
                                        y=viajes['DURACION_MIN'][viajes['VICTIMA_AGRESION'] == '1'][viajes['DURACION_MIN']<300],
                                        mode='markers',
                                        text=viajes['ESTRATO_y'][viajes['VICTIMA_AGRESION'] == '1'][viajes['DURACION_MIN']<300],
                                        ##opacity=0.7,
                                        marker={
                                            'color': viajes['ESTRATO_y'][viajes['VICTIMA_AGRESION'] == '1'][viajes['DURACION_MIN']<300],
                                            'size': 15,
                                            'line': {'width': 0.5, 'color': 'white'}
                                        },
                                    
                                        ) 
                                        ],
                                        'layout': go.Layout(
                                            xaxis=dict(dtick=1,gridcolor='rgb(255, 255, 255)',title='HORA DE INICIO DEL VIAJE (HH)'),
                                            yaxis=dict(dtick=30,gridcolor='rgb(255, 255, 255)',title='DURACION DEL VIAJE (MIN)'),
                                            hovermode='closest', title='Agresiones según hora de viaje y su duración'
                                        )
                                    }
                                ),
     ],style={'width': '95%', 'display': 'inline-block'}
     ),##cierre bloque A                    
     html.Div([##bloqueB
             #html.Div(id='output-container-button', children='Ingrese la siguiente información'),
             html.Label('Ingrese la siguiente información'),
             html.Div([##cuad1        
                    html.Label('Digite su edad:'),
                    dcc.Input( id='Edad',
                    min=viajes['EDAD'].min(),
                    max=viajes['EDAD'].max(),
                    value=25, type='integer'),
                              
                    html.Label('Seleccione su género:'),
                    dcc.Dropdown(id='Genero',
                        options=[
                            {'label': 'Masculino', 'value': 'Hombre'},
                            {'label': 'Femenino', 'value': 'Mujer'}
                        ],
                       value='Mujer',
                       style={'width': '50%'}
                       ),
                    
                    html.Label('Seleccione su Estrato:'),
                    dcc.RadioItems(id='Estrato',
                            options=[{'label': i, 'value': i} for i in sorted(viajes['ESTRATO_y'].unique())],
                            value=2,
                            labelStyle={'display': 'inline-block'}
                        ),
                    
                    html.Label('Digite el tiempo promedio del viaje (minutos):'),
                    dcc.Input( id='minutos',
                    min=viajes['DURACION_MIN'].min(),
                    max=viajes['DURACION_MIN'].max(),
                    value=40, type='integer'),
                              
                    html.Label('Seleccione el número de etapas:'),
                    dcc.Dropdown(id='etapas',
                            options=[{'label': i, 'value': i} for i in sorted(viajes['CANTIDAD_ETAPAS'].unique())],
                        value=2
                    ,style={'margin':'auto','float':'left','width': '60%', 'display': 'inline-block'}
                    ),
                    
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
                    dcc.Input( type='text', id = 'direccion_origen',value='calle 130a#59c-50, bogota'),               
                            
                    html.Label('Seleccione destino:'),
                    dcc.Input( type='text', id = 'direccion_destino',value='carrera 88a#21-75, bogota'),
                   

                    html.Label('Seleccione la hora de viaje:'),          
                        dcc.Slider(id='hora',
                        min=viajes['HORA_DEC'].min(),
                        max=viajes['HORA_DEC'].max(),
                        marks={str(hora): str(hora) for hora in viajes['HORA_DEC'].unique()},
                        value=15
                    ),
            ],style={'width': '50%', 'display': 'inline-block'}
            )##cierre cuad2
    ]),##cierre bloque B
    html.Div([
           html.Label('Presione ENVIAR para cargar su información'), 
            html.Button('ENVIAR', id='button',style={'margin':'auto','width': "30%", 'backgroundColor': 'grey'})
    ]),
    html.Div([#BloqueC
                    html.Div([ ##cuad3.1
                          dcc.Graph(
    						id='bar_plot',      
    								figure= go.Figure(data=[go.Bar(x=['Probabilidad'],y=[probab],name='Probabilidad'), 
    														go.Bar(x=['Probabilidad'],y=[1-probab],name='')],
    								   layout=go.Layout(barmode='stack', title='Probabilidad de agresión'),
    								   )
    								),                                  
                        html.Div([
                            html.Label('Según los parametros ingresados, la probabilidad de ser agredido es de:'),
                            html.Div(id='prob_text', style={'color': 'blue', 'fontSize': 30}),
                        ], style={'marginBottom': 50, 'marginTop': 25,'fontSize': 18}),
 ##                      html.H1(children=probab*100),
                    ],style={'width': '20%', 'display': 'inline-block','fontSize': 18}
                    ), ##cierre cuad3.1
                        
                    html.Div([ ##cuad4.2   Esta es la seccion del mapa
                         dcc.Graph(id='main_map',figure=figureIni),
                        ],style={'height': '85vh','width': '80%', 'display': 'inline-block'}                                
                    ), ##cierre cuad4.2 """                    
    ]),#CierreBloqueC
    html.Div(id = 'hidden_coords')                            
]
)
                                    

@app.callback(dash.dependencies.Output('hidden_coords', 'children'),[],
              [dash.dependencies.State('direccion_origen','value'),
               dash.dependencies.State('direccion_destino','value')],
              [dash.dependencies.Event('button','click')])


def update_Coordinates(direccion_origen,direccion_destino):
    ######################### ArcGIS ##################
    resultsOrigen = []
    resultsOrigen = geocode(direccion_origen)
    posicionOrigen = resultsOrigen[0]['location']
    resultsDestino = []
    resultsDestino = geocode(direccion_destino)
    posicionDestino = resultsDestino[0]['location']
    
    
    global lat_origen1
    lat_origen1=posicionOrigen['y']
    global long_origen1
    long_origen1=posicionOrigen['x']
    global lat_destino1
    lat_destino1=posicionDestino['y']
    global long_destino1
    long_destino1=posicionDestino['x']
    print(lat_origen1, "Lat-Orig")
    print(long_origen1, "Lon-Orig")
    print(lat_destino1, "Lat-Dest")
    print(long_destino1, "Lon-Dest")
    ##################################################
    return html.Div([dcc.Input(value=posicionOrigen['y'], id = 'lat_origen', type = 'hidden'),
                     dcc.Input(value=posicionOrigen['x'], id = 'long_origen', type = 'hidden'),
                     dcc.Input(value=posicionDestino['y'], id = 'lat_destino', type = 'hidden'),
                     dcc.Input(value=posicionDestino['x'], id = 'long_destino', type = 'hidden')])

    
@app.callback(dash.dependencies.Output('main_map','figure'),
              [dash.dependencies.Input('lat_origen','value'),
               dash.dependencies.Input('long_origen','value'),
               dash.dependencies.Input('lat_destino','value'),
               dash.dependencies.Input('long_destino','value')])
    
def update_route(lat_origen,long_origen,lat_destino,long_destino):
    origen = (long_origen,lat_origen)
    destino = (long_destino,lat_destino)
    response = serviceRoutes.directions([origen,destino], 'mapbox/driving', geometries = 'geojson')
    global rJson
    rJson = response.json()
    ruta = rJson['routes'][0]['geometry']
    latitudes = [lat_origen,lat_destino]
    longitudes = [long_origen,long_destino]
    texto = ['Origen','Destino']
    dataMap = [
    go.Scattermapbox(showlegend = False,
        lat=latitudes,
        lon=longitudes,
        mode='markers',
        marker=dict(
            size=10,####Tamaño de los marcadores
            color='rgb(200,0,0)'
        ),
        text=texto,
        )]
    ###El elemento layers pinta la ruta
    layers=[dict(sourcetype = 'geojson',
         source = ruta,
         color='rgb(0,0,200)',
         type = 'line',
         line=dict(width=2)
         )]
    mapLayout = go.Layout(
        autosize=True,
        height = 400,
        hovermode='closest',
        title='Mapa con Ruta',
        font=dict(color='#000101'),
        titlefont=dict(color='#000101', size=14),
        margin=dict(l=35,r=35,b=30,t=30), ### Sirve para ajustar el mapa
        paper_bgcolor="#57A1E6", ####Si cambiaron el fondo arriba aqui tambien con el mismo color
        mapbox=dict(
            accesstoken=mapbox_access_token,
            style="basic",
            layers = layers,
            center=dict(
                lat=lat_origen, ###Estas coordenadas centran el mapa
                lon=long_origen
            ),
            pitch=0,
            zoom=9 ###El zoom de acercamiento al mapa
        ))
    mapa_con_ruta = dict(data=dataMap, layout=mapLayout)
    return mapa_con_ruta
   


  
                
@app.callback(
     dash.dependencies.Output('bar_plot', 'figure'),
  #  dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('lat_origen','value'),
     dash.dependencies.Input('long_origen','value'),
     dash.dependencies.Input('lat_destino','value'),
     dash.dependencies.Input('long_destino','value')],
    [dash.dependencies.State('Edad', 'value'),
     dash.dependencies.State('Genero', 'value'),
     dash.dependencies.State('Estrato', 'value'),
     dash.dependencies.State('transporte', 'value'),
     dash.dependencies.State('dia', 'value'),
     dash.dependencies.State('hora', 'value'),
     dash.dependencies.State('etapas', 'value'),
     dash.dependencies.State('minutos', 'value')
    ])
def update_graph(lat_origen,long_origen,lat_destino,long_destino,   Edad,Genero,Estrato,transporte,dia,hora,etapas,minutos):
    
 
    ex_dic = {
            'MEDIO_PREDOMINANTE':transporte,
            'ZAT_DESTINO':zat_destino,
            'ZAT_ORIGEN':zat_orig,
            'HABIL':dia,
            'DISTANCIA': Distance(df_zats.loc[df_zats['ZAT_ORIGEN']==int(zat_orig),'Latitud_Origen'].iloc[0],df_zats.loc[df_zats['ZAT_ORIGEN']==int(zat_orig),'Longitud_Origen'].iloc[0],df_zats.loc[df_zats['ZAT_DESTINO']==int(zat_destino),'Latitud_Destino'].iloc[0],df_zats.loc[df_zats['ZAT_DESTINO']==int(zat_destino),'Longitud_Destino'].iloc[0]),
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
    probab = pd.DataFrame( model.predict_proba(input_df2))[1]

    probab=probab.iloc[0]
    print(probab) 
    
    return  {'data': [go.Bar( x=['Probabilidad'], y=[probab],name='Probabilidad'),go.Bar(x=['Probabilidad'],y=[0.35-(0.35*probab)],name='')],
             'layout':go.Layout(barmode='stack', title='Probabilidad de agresión')
             }



@app.callback(
     dash.dependencies.Output('prob_text', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('Edad', 'value'),
     dash.dependencies.State('Genero', 'value'),
     dash.dependencies.State('Estrato', 'value'),
     dash.dependencies.State('transporte', 'value'),
     dash.dependencies.State('dia', 'value'),
     dash.dependencies.State('hora', 'value'),
     dash.dependencies.State('etapas', 'value'),
     dash.dependencies.State('minutos', 'value')
    ])
def update_text(n_clicks, Edad,Genero,Estrato,transporte,dia,hora,etapas,minutos):
    
    global zat_orig
    zat_orig=(closest_orig(df_zats,lat_origen1,long_origen1))
    df_zats_orig=df_zats[df_zats['ZAT_ORIGEN']==zat_orig]
    global zat_destino
    zat_destino=(closest_dest(df_zats_orig,lat_destino1,long_destino1))
    
    ex_dic = {
            'MEDIO_PREDOMINANTE':transporte,
            'ZAT_DESTINO':zat_destino,
            'ZAT_ORIGEN':zat_orig,
            'HABIL':dia,
            'DISTANCIA': Distance(df_zats.loc[df_zats['ZAT_ORIGEN']==int(zat_orig),'Latitud_Origen'].iloc[0],df_zats.loc[df_zats['ZAT_ORIGEN']==int(zat_orig),'Longitud_Origen'].iloc[0],df_zats.loc[df_zats['ZAT_DESTINO']==int(zat_destino),'Latitud_Destino'].iloc[0],df_zats.loc[df_zats['ZAT_DESTINO']==int(zat_destino),'Longitud_Destino'].iloc[0]),
            'CANTIDAD_ETAPAS': etapas,
            'DURACION_MIN': minutos,
            'SEXO': Genero,
            'EDAD': Edad,
            'ESTRATO_y': Estrato,
            'HORA_DEC': hora } 
    print(ex_dic)

    input_df = pd.DataFrame( ex_dic, index=[150000])
    viajes4=viajes3.append(input_df,ignore_index=False)
    input_df_dummy = pd.get_dummies(viajes4, columns=['MEDIO_PREDOMINANTE','ZAT_DESTINO','ZAT_ORIGEN','SEXO','ESTRATO_y'])
    input_df2 = pd.DataFrame( input_df_dummy.loc[[150000]])
    probab = pd.DataFrame( model.predict_proba(input_df2))[1]
    halfwidth = abs(model.decision_function(input_df2)/100)
    intervalo_inf = probab - (halfwidth/2)
    intervalo_sup = probab + (halfwidth/2)


    probab=probab.iloc[0]*100
    intervalo_inf=intervalo_inf.iloc[0]*100
    intervalo_sup=intervalo_sup.iloc[0]*100
    return "["+ str(round(intervalo_inf,3))+" %, "+str(round(intervalo_sup,3))+" %]"

   # return str(round(probab,3))+" %"
      
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
    
