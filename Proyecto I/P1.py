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
import numpy as np
import category_encoders as ce
import pickle
from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
#from sklearn import metrics
#from sklearn.metrics import mean_squared_error
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor


# =============================================================================
# CARGA DE DATASET
# train1 = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTrain_carListings.zip')
# test1 =  pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/dataTest_carListings.zip', index_col=0)
# train1['source']='train'
# test1['source']='test'
# data = pd.concat([train1, test1],sort=True)
# data.to_pickle("df_price_car.pkl")
# =============================================================================
data = pd.read_pickle("df_price_car.pkl")

# =============================================================================
# CREACION ENCODER
# encoder = ce.BinaryEncoder(cols=['Model','State','Make'])
# encoder.fit_transform(data)
# data=encoder.fit_transform(data)
# pickle.dump(encoder, open( 'encoder.sav', "wb" ) )
# =============================================================================
encoder = pickle.load( open( 'encoder.sav', "rb" ) )

# =============================================================================
# ENTRENAMIENTO MODELO
# train = data.loc[data['source']=="train"]
# test = data.loc[data['source']=="test"]
# #Drop unnecessary columns:
# test.drop(['Price','source'],axis=1,inplace=True)
# train.drop(['source'],axis=1,inplace=True)
# ### X e y, definición de  train y test para entremaiento
# X = train.drop(['Price'], axis=1)
# y = train['Price']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)
# ## Entrenamiento modelo 
# linreg=LinearRegression()
# linreg.fit(X_train, y_train)
# ### Save model
# pickle.dump(linreg,open('model.sav', 'wb'))
# 
# =============================================================================
model = pickle.load(open('model.sav', 'rb'))


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions']=True # Sirve para evitar errores al inicio de callbacks 
                                                # que dependan de elementos no creados al momento de

app.layout = html.Div(children=[
    html.Div([
        html.Div([##bloqueA
            html.H1(children='PRICING',style={'margin':'auto','width': "50%",'color': 'brown', 'fontSize': 40}),
            html.Div(children='''
                Aplicación para el cálculo del precio aproximado de un auto usado
            ''',style={'margin':'auto','width': "70%",'color': 'brown', 'fontSize': 20}),
        ]),
    ],style={'width': '100%', 'display': 'inline-block','color': 'brown', 'fontSize': 40}),
     html.Div([##bloqueA
             
                  
                html.Label('Seleccione el año:'),
                dcc.Dropdown(id='Year',
                        options=[{'label': i, 'value': i} for i in sorted(data['Year'].unique())],
                    value=2015
                ,style={'margin':'auto','float':'left','width': '40%', 'display': 'inline-block'}
                ),
                
                html.Label('Seleccione las millas'),          
                        dcc.Input(id='Mileage',
                        min=data['Mileage'].min(),
                        max=data['Mileage'].max(),
                        value=55076, type='float'),
                                   
                html.Label('Seleccione el Estado:'),
                dcc.Dropdown(id='State',
                        options=[{'label': i, 'value': i} for i in sorted(data['State'].unique())],
                    value='MD'
                ,style={'margin':'auto','float':'left','width': '40%', 'display': 'inline-block'}
                ),
                       
                       
                html.Label('Seleccione el fabricante:'),
                dcc.Dropdown(id='Make',
                        options=[{'label': i, 'value': i} for i in sorted(data['Make'].unique())],
                    value='Nissan'
                ,style={'margin':'auto','float':'left','width': '40%', 'display': 'inline-block'}
                ),
                       
                html.Label('Seleccione el Modelo:'),
                dcc.Dropdown(id='Model',
                        options=[{'label': i, 'value': i} for i in sorted(data['Model'].unique())],
                    value='MuranoAWD'
                ,style={'margin':'auto','float':'left','width': '40%', 'display': 'inline-block'}
                ),
                
                    
                ],style={'margin':'auto','width': '50%', 'display': 'inline-block','backgroundColor': 'white'}

           ),##cierre bloque A 
                   
     html.Div([##bloqueB
            html.Div([##cuad1  
                 
                    
       ],style={'margin':'auto','width': '50%', 'display': 'inline-block'}
            ), ##cierre cuad1        
            html.Div([ ##cuad2        

            ],style={'width': '50%', 'display': 'inline-block'}
            )##cierre cuad2
    ]),##cierre bloque B
    
    html.Div([
           html.Label('Presione ENVIAR para cargar su información'), 
            html.Button('ENVIAR', id='button',style={'margin':'auto','width': "30%", 'backgroundColor': 'grey'})
    ]),
            
        html.Div([#BloqueC
                        html.Div([ ##cuad3.1
                                     
                            html.Div([
                                html.Label('Según los parametros ingresados, el precio aproximado del auto usado es de:'),
                                html.Div(id='price_text', style={'color': 'blue', 'fontSize': 30}),
                            ], style={'marginBottom': 50, 'marginTop': 25,'fontSize': 18}),
     ##                      html.H1(children=probab*100),
                        ],style={'width': '20%', 'display': 'inline-block','fontSize': 18}
                        ), ##cierre cuad3.1
                            
                        html.Div([ ##cuad4.2   
    
    
                            ],style={'height': '85vh','width': '80%', 'display': 'inline-block'}                                
                        ), ##cierre cuad4.2 """                    
        ]),#CierreBloqueC
    html.Div(id = 'hidden_coords')                            
 ])

@app.callback(
     dash.dependencies.Output('price_text', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('Year', 'value'),
     dash.dependencies.State('Mileage', 'value'),
     dash.dependencies.State('State', 'value'),
     dash.dependencies.State('Make', 'value'),
     dash.dependencies.State('Model', 'value')]
     )
 

def update_text(n_clicks, Year, Mileage, State, Make, Model):   
     ex_dic = {

            'Mileage':Mileage,
            'Price':0,
            'Year':Year ,
            'source':'test',
            'State':State,
            'Make':Make,
            'Model': Model
            
            }

     input_df = pd.DataFrame( ex_dic, index=[1])
     input_df = encoder.transform(input_df)
     price = model.predict(input_df.drop(['Price','source'],axis=1))
     print(ex_dic)
     print(price)
     return str(np.round(price,2))
     

   # return str(round(price,3))+" %"
      
if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)