import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction
import dash.dependencies as dd
from importlib_resources import path
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from datetime import datetime as dt
import pathlib
import random

import nltk
import spacy
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

from wordcloud import WordCloud
import base64
from io import BytesIO


# import en_core_web_sm
# nlp = en_core_web_sm.load()
nlp = spacy.load("en_core_web_sm")


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Onto Dashboard"

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data


df = pd.read_csv(DATA_PATH.joinpath("topic-6/version.csv"))
version_list = [6,9,14,17]
doc_list = [i for i in range(10000)]
topic_list = df['Dominant_Topic'].unique()
df["Admit Source"] = df["speaker"].fillna("Unknown")
speaker_name = df["Admit Source"].unique().tolist()
speakername_1 = [{'label':k,'value':k} for k in speaker_name]
speakername_2 = [{'label':'select all','value':'all'}]
speakeroption = speakername_1 + speakername_2

df['party_name'] = df['party'].fillna('Unknown')
party_name = df['party_name'].unique().tolist()
partyoption = [{"label": i, "value": i} for i in party_name]

df['city'] = df['location'].fillna('Unknown')
city_name = df['city'].unique().tolist()
cityname_1 = [{'label':k,'value':k} for k in city_name]
cityname_2 = [{'label':'select all','value':'all'}]
cityoption = cityname_1 + cityname_2

print(df.columns)
layout_list = ['spring layout','planar layout','circular layout','spectral layout','spiral layout']
layoutoption = [{'label':k,'value':k} for k in layout_list]
parameter =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
paremeteroption =  [{'label':k,'value':k} for k in parameter]

#for parallel coordinate

df_p = df[['doc','topic1',"topic2",'topic3','topic4','topic5','topic6','Dominant_Topic']]
dimension = df_p.columns.values.tolist()
dimension.pop()
#for word_cloud

df_topic_keywords = pd.read_csv(DATA_PATH.joinpath("topic-6/keyweight_30.csv"))

#graph visualisation text test
text = df['speech'][8]

# def draw_graph(data):
#     def plot_associations(text, k=0.3, font_size=26):

#         nouns_in_text = []
#         is_noun = lambda pos: pos[:2] == 'NN'

#         for sent in text.split('.')[:-1]:   
#             tokenized = nltk.word_tokenize(sent)
#             nouns=[word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
#             nouns_in_text.append(' '.join([word for word in nouns if not (word=='' or len(word)==1)]))

#         nouns_list = []

#         for sent in nouns_in_text:
#             temp = sent.split(' ')
#             for word in temp:
#                 if word not in nouns_list:
#                     nouns_list.append(word)

#         df = pd.DataFrame(np.zeros(shape=(len(nouns_list),2)), columns=['Nouns', 'Verbs & Pres'])
#         df['Nouns'] = nouns_list

#         is_adjective_or_verb = lambda pos: pos[:2]=='JJ' or pos[:2]=='VB' or pos[:2] == 'IN'
#         for sent in text.split('.'):
#             for noun in nouns_list:
#                 if noun in sent:
#                     tokenized = nltk.word_tokenize(sent)
#                     adjectives_or_verbs = [word for (word, pos) in nltk.pos_tag(tokenized) if is_adjective_or_verb(pos)]
#                     ind = df[df['Nouns']==noun].index[0]
#                     df['Verbs & Pres'][ind]=adjectives_or_verbs
   
#         return df
#     df = plot_associations(data)
    
#     gg = nx.Graph()
# #G 这个图
#     for i in range(len(df)):
#         gg.add_node(df['Nouns'][i])
#         for word in df['Verbs & Pres'][i]:
#             gg.add_edges_from([(df['Nouns'][i], word)])
#     pos = nx.spring_layout(gg,0.5)
    
#     color_list = []
#     N = []
#     for i in gg.nodes:
        
#         value = nltk.pos_tag([i])[0][1]

#         #print(value)
#         if (value=='NN' or value=='NNP' or value=='NNS'):
#             color_list.append('#ccccff')
#             N.append('NN')

#         elif value =='IN':
#             color_list.append('#ccffcc')
#             N.append('IN')

#         else:
#             color_list.append('#ffcccc')
#             N.append('Verb')
#     edge_trace = go.Scatter(
#         x=[],y=[],
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='none',
#         mode='lines')
#     for edge in gg.edges():
#         x0,y0=pos.get(edge[0])
#         x1,y1=pos.get(edge[1])
#         edge_trace['x']+=tuple([x0,x1,None])
#         edge_trace['y']+=tuple([y0,y1,None])
#     node_trace = go.Scatter(
#         x=[],y=[],
#         hoverinfo='text',
#         mode="markers+text",
#         name="Entity",
#         text= list(pos.keys()),
#         marker=dict(size=10, color=color_list))    

#     for node in gg.nodes():
#         x, y=pos.get(node)
#         node_trace['x']+=tuple([x])
#         node_trace['y']+=tuple([y])
        
#     fig = go.Figure(data=[edge_trace, node_trace],
#              layout=go.Layout(
#                 # title='<br>Network graph made with Python',
#                 titlefont_size=16,
#                 width=400,
#                 height=500,
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                 )
#     return fig



def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Ontology Analytics"),
           # html.H3(" Ontology Analytics Dashboard"),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Topic Version"),
            dcc.Dropdown(
                id="topic-select_version",
                options=[{"label": i, "value": i} for i in version_list],
                value=version_list[0],
            ),
            html.Br(),
            html.P("Select Time"),
            dcc.RangeSlider(1979, 2021, 1,value=[1979, 2021],marks={
                            1979:{'label':'1979'},
                            1986:{'label':'1896'},
                            1993:{'label':'1993'},
                            2000:{'label':'2000'},
                            2007:{'label':'2007'},
                            2014:{'label':'2014'},
                            2021:{'label':'2021'},},  id='my-range-slider'),
            html.Br(),
            html.P("Select Speaker"),
            dcc.Dropdown(
                id="speaker-select",
                options=speakeroption,
                value='all',
                multi=True,
            ),
            html.Br(),
            html.P("Select Party"),
            dcc.Dropdown(
                id="party-select",
                options=partyoption,
                value=party_name[:],
                multi=True,
            ),
            html.Br(),
            html.P("Select City"),
            dcc.Dropdown(
                id="city-select",
                options=cityoption,
                value='all',
                multi=True,
            ),
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="select-all", children="Reset", n_clicks=0),
            ),

        ],
    )
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}






#layout 

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children = [html.Img(src=app.get_asset_url("logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="three columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none",'width':"20%"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # topic-vis
                html.Div(
                    id="topic-vis_card",
                    children=[
                        html.B("Topic Visulisation"),
                        html.Hr(),
                        html.Div(id='page',children=[
                            html.Iframe(src="assets/ldamallet.html",style={'scrolling':'no'}),],
                            style={'width': '68%','display':'inline-block'}),
                        
                        html.Div(id='word-vis',children=[
                            html.P("Select Topic:",style={'display':'inline-block','float':'left','margin-right':'15px','margin-top':'13px'}),
                            dcc.Dropdown(
                            id="topic-id",
                            options=topic_list,
                            value=topic_list[0],),
                            html.P('Keywords distribution per topic',style={'margin-top':'18px','margin-left':'100px'}),
                            html.Img(id='word-cloud',)
                            ],
                            style={'display':'inline-block',"float":"right",'width':"30%"})
                        ],

                ),
                #scatter plot and graph
                html.Div(
                    id="patient_volume_card",
                    children=[
                        html.B("Ontology visulisation"),
                        html.Hr(),
                        html.Div(
                            id="scatter plot",
                            children=[
                                html.Div([
                                dcc.Graph(id ='scatter')]),
                            ],style={'width': '58%', 'display': 'inline-block', 'padding': '0 10'},
                            ),
                        html.Div(
                            id = 'relation plot',
                            children =[
                                html.Div([
                                    html.Div([
                                        dcc.Dropdown(id="docid", options=doc_list,value=5),
                                        ],style={'display':'inline-block','width':'35%'}),
                                    html.Div([
                                        dcc.Dropdown(id="layout",options=layoutoption,value=layout_list[0],multi=False),
                                    ],style={'display':'inlin-block','width':'35%','float': 'right','margin-right':'30px'}),
                                    html.Div([
                                        dcc.Graph(id='relation-graph')
                                    ])
                                    ,])],
                                    style={'display': 'inline-block', 'float':'right' ,'width': '40%'}
                            ),
                        # html.Div(className='row',children=[
                        #     html.Pre(id ='hover-data',style=styles['pre'])
                        # ]),
                        
                    ],
                ),
                # parallel coordinate
                html.Div(
                    id="wait_time_card",
                    children=[
                        html.B("parallelcoordinate"),
                        html.Hr(),
                        html.Br(),
                        html.Div([
                        dcc.Slider(id='slider',min=0, max=10000,step=100, marks=None, value=10000)],
                                  style={'width': '48%', 'display': 'inline-block'}),
                        html.Div([ 
                        dcc.Input(id="searchid", type="number", placeholder="input doc",min=0, max=10000, step=1,),],
                                 style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
                        html.Div([
                        dcc.Graph(id='parallel')]),
                    ],
                ),
               
            ],
        ),
    ],
)


# call back function for interaction of graphs

# parallel graph interaction
@app.callback(Output('parallel','figure'),Input('slider','value'),Input('searchid','value'),Input('scatter', 'selectedData'))

def parallel_chart(step,rangeval,selectedData):
    if rangeval:
        step = 0
        dff = df_p[df_p['doc']==rangeval]
    if step:
        dff = df_p[:step]
    if selectedData:
        points_selected = []
        for point in selectedData['points']:
            points_selected.append(point['pointIndex'])
        dff = df_p[df_p['doc'].isin(points_selected)]
    if step and selectedData:
        points_selected = []
        for point in selectedData['points']:
            points_selected.append(point['pointIndex'])
        df_parallel = df_p[:step]
        dff = df_parallel[df_parallel['doc'].isin(points_selected)]
       
    fig_parallel = go.Figure(data=
        go.Parcoords(
            line = dict(color = dff["Dominant_Topic"],
                     colorscale=[[0,'red'],[0.2,'orange'],[0.4,'yellow'],[0.6,'green'],[0.8,'blue'],[1,'purple']]),
                    
            dimensions = list([
                dict(range = [0,10000],
                    constraintrange = [0,2],
                    label = 'DOC ID', values = dff['doc']),
                dict(range = [0,1],
                    label = 'Topic 1', values = dff['topic1']),
                dict(range = [0,1],
                    label = 'Topic 2', values = dff['topic2']),
                dict(range = [0,1],
                    label = 'Topic 3', values = dff['topic3']),
                dict(range = [0,1],
                    label = 'Topic 4', values = dff['topic4']),
                dict(range = [0,1],
                    label = 'Topic 5', values = dff['topic5']),
                dict(range = [0,1],
                    label = 'Topic 6', values = dff['topic6'])]))
                )
        
    return fig_parallel


 # scatter filter           
@app.callback(
    Output('scatter', 'figure'),
    [Input('my-range-slider', 'value'),
    Input('party-select','value'),
    Input('speaker-select','value'),
    Input('city-select','value')])
def update_scatter(value,partyselect,speakerselect,cityselect):
    if value[0] == value[1]:
        result = 'year: {0[0]}'.format(value)
    else:
        result = 'year: {0[0]} - {0[1]}'.format(value)
    
    df_filter_time = df.loc[(df['clean_date'] >= value[0]) & (df['clean_date'] <= value[1])]
    df_filter_party = df_filter_time.loc[df_filter_time['party'].isin(partyselect)]
    #dropdown_speaker = speakerselect
    if 'all' in speakerselect:
      #  dropdown_speaker = df['speaker']
        df_filter_speaker = df_filter_party
    else:
        df_filter_speaker = df_filter_party.loc[df_filter_party['speaker'].isin(speakerselect)]
    if 'all' in cityselect:
        df_filter_city = df_filter_speaker
    else:
        df_filter_city = df_filter_speaker.loc[df_filter_speaker['location'].isin(cityselect)]
        
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x= df_filter_city.loc[df_filter_city['Dominant_Topic']==1]['x'],
        y = df_filter_city.loc[df_filter_city['Dominant_Topic']==1]['y'],
        text=df_filter_city.loc[df_filter_city['Dominant_Topic']==1]['doc'],
        name = 'topic1',
        mode = 'markers',
        customdata=df_filter_city[['Text','ref', 'speaker','party','clean_date','location']],
        marker = dict(color = '#FF0000'),
        # hovertemplate="<b>%{text}</b><br>"+"ref:%{customerdata}</b><br>",
        unselected={'marker': {'opacity': 0.3}}

    ))
    fig_scatter.add_trace(go.Scatter(
        x= df_filter_city.loc[df_filter_city['Dominant_Topic']==2]['x'],
        y = df_filter_city.loc[df_filter_city['Dominant_Topic']==2]['y'],
        text=df_filter_city.loc[df_filter_city['Dominant_Topic']==2]['doc'],
        name = 'topic2',
        mode = 'markers',
        customdata=df_filter_city[['Text','ref', 'speaker','party','clean_date','location']],
        marker = dict(color = '#F15A22'),
        # hovertemplate="<b>%{text}</b><br>"+"ref:%{customerdata}</b><br>",
        unselected={'marker': {'opacity': 0.3}}

    ))
    fig_scatter.add_trace(go.Scatter(
        x= df_filter_city.loc[df_filter_city['Dominant_Topic']==3]['x'],
        y = df_filter_city.loc[df_filter_city['Dominant_Topic']==3]['y'],
        text=df_filter_city.loc[df_filter_city['Dominant_Topic']==3]['doc'],
        name = 'topic3',
        mode = 'markers',
        customdata=df_filter_city[['Text','ref', 'speaker','party','clean_date','location']],
        marker = dict(color = '#FFEF00'),
        # hovertemplate="<b>%{text}</b><br>"+"ref:%{customerdata}</b><br>",
        unselected={'marker': {'opacity': 0.3}}

    ))
    fig_scatter.add_trace(go.Scatter(
        x= df_filter_city.loc[df_filter_city['Dominant_Topic']==4]['x'],
        y = df_filter_city.loc[df_filter_city['Dominant_Topic']==4]['y'],
        text=df_filter_city.loc[df_filter_city['Dominant_Topic']==4]['doc'],
        name = 'topic4',
        mode = 'markers',
        customdata=df_filter_city[['Text','ref', 'speaker','party','clean_date','location']],
        marker = dict(color = '#00FF00'),
        # hovertemplate="<b>%{text}</b><br>"+"ref:%{customerdata}</b><br>",
        unselected={'marker': {'opacity': 0.3}}

    ))
    fig_scatter.add_trace(go.Scatter(
        x= df_filter_city.loc[df_filter_city['Dominant_Topic']==5]['x'],
        y = df_filter_city.loc[df_filter_city['Dominant_Topic']==5]['y'],
        text=df_filter_city.loc[df_filter_city['Dominant_Topic']==5]['doc'],
        name = 'topic5',
        mode = 'markers',
        customdata=df_filter_city[['Text','ref', 'speaker','party','clean_date','location']],
        marker = dict(color = '#0000FF'),
        # hovertemplate="<b>%{text}</b><br>"+"ref:%{customerdata}</b><br>",
        unselected={'marker': {'opacity': 0.3}}

    ))
    fig_scatter.add_trace(go.Scatter(
        x= df_filter_city.loc[df_filter_city['Dominant_Topic']==6]['x'],
        y = df_filter_city.loc[df_filter_city['Dominant_Topic']==6]['y'],
        text=df_filter_city.loc[df_filter_city['Dominant_Topic']==6]['doc'],
        name = 'topic6',
        mode = 'markers',
        customdata=df_filter_city[['Text','ref', 'speaker','party','clean_date','location']],
        marker = dict(color = '#800080'),
        # hovertemplate="<b>%{text}</b><br>"+"ref:%{customerdata}</b><br>",
        unselected={'marker': {'opacity': 0.3}}

    ))
    fig_scatter.update_traces(
        hovertemplate="<br>".join([ 
                                   "doc:%{text}",
                                   "text:%{customdata[0]}",
                                   "ref: %{customdata[1]}",
                                   "author:%{customdata[2]}",
                                   "party:%{customdata[3]}"])

    )
    fig_scatter.update_layout(
                    title = result,
                    plot_bgcolor = 'white',
                    dragmode='lasso', 
                    hovermode = 'closest',
                    uirevision = 'constant',
                    )
    return fig_scatter
    
@app.callback(Output('relation-graph','figure'),Input('scatter', 'hoverData'), Input('docid','value'),Input('layout','value'))
def update_relation(hoverData,docid,layout):
    if hoverData:
        id = hoverData['points'][0]['text']
        f_data = df['relation'][id] 
    if docid:
        f_data = df['relation'][docid]
    f = eval(f_data)
    print(f)
    gg = nx.Graph()
    for triple in f:
        gg.add_node(triple[0])
        gg.add_node(triple[1])
        gg.add_node(triple[2])
        gg.add_edge(triple[0], triple[1])
        gg.add_edge(triple[1], triple[2])
        #layout_list = ['spring layout','planar layout','circular layout','spectral layout','spiral layout','kamada_kawai layout']
    if layout =='planar layout':
        pos = nx.planar_layout(gg,0.8)
    elif layout =='circular layout':
        pos = nx.circular_layout(gg,0.8)
    elif layout == 'spectral layout':
        pos =nx.spectral_layout(gg,0.8)
    elif layout == 'spiral layout':
        pos = nx.spiral_layout(gg,0.8)
    else:
        pos = nx.spring_layout(gg,0.8)
    color_list = []
    N = []
    for i in gg.nodes:
        doc = nlp(i)
        if doc[0].pos_ == 'prep'or doc[0].pos_=='ADP':
            color_list.append('#ccccff')
            N.append('prep')
    #         doc_list.append(i)
        elif doc[0].pos_ == 'AUX' or doc[0].pos_ == 'VERB'or doc[0].pos_ =='INTJ':
            color_list.append('#ccffcc')
            N.append('Verb')
    #         doc_list.append(i)
        else:
            color_list.append('#ffcccc')
            N.append('Noun')
    #         doc_list.append(i)
    edge_trace = go.Scatter(
        x=[],y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    for edge in gg.edges():
        x0,y0=pos.get(edge[0])
        x1,y1=pos.get(edge[1])
        edge_trace['x']+=tuple([x0,x1,None])
        edge_trace['y']+=tuple([y0,y1,None])
    node_trace = go.Scatter(
        x=[],y=[],
        hoverinfo='text',
        mode="markers+text",
        name="Entity",
        text= list(pos.keys()),
        marker=dict(size=10, color=color_list))    

    for node in gg.nodes():
        x, y=pos.get(node)
        node_trace['x']+=tuple([x])
        node_trace['y']+=tuple([y])

    fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                # title='<br>Network graph made with Python',
                titlefont_size=16,
                width=550,
                height=410,
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=5,l=5,r=5,t=4),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig


@app.callback(Output('my-range-slider','value'),
            Output('speaker-select','value'),
            Output('city-select','value'), 
            Output('party-select','value'),
            Input("select-all", "n_clicks"))
def select_all(n_clicks):
    
    slider = [1979,2021]
    speaker = [option["value"] for option in speakername_2]
    city = [option["value"] for option in cityname_2]
    party = [option["value"] for option in partyoption]
    return slider,speaker,city,party

#scatter plot and graph interactive visualisation
# @app.callback(
#     Output('hover-data', 'children'),
#     Input('scatter', 'hoverData'))
# def display_hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)

@app.callback(Output('word-cloud', 'src'), [Input('topic-id', 'value')])
def draw_img(id):
    def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
        return "hsl(208, 58%%, %65d%%)" % random.randint(45, 60)
    df_topic_0 = df_topic_keywords[df_topic_keywords['topic']==id]
    df0 = df_topic_0.set_index(['keywords'])['fre'].to_dict()
    wc = WordCloud(background_color='white').generate_from_frequencies(frequencies=df0).recolor(color_func=grey_color_func, random_state=5)
    wc_img = wc.to_image()
    with BytesIO() as buffer:
        wc_img.save(buffer, 'png')
        img2 = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," +img2


# Run the server
if __name__ == "__main__":  
   app.run_server(debug=True)
