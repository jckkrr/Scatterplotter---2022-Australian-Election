############## Constistuent.Online #################
####### Code-free analysis for curious folk. ######

### An application for ...

## streamlit run "C:\Users\Jack\Documents\Python_projects\streamlit_apps\parli_scatterplotter\streamlit_app.py"

### --------------------------------------- IMPORTS 

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import customChartDefaultStyling

pd.set_option('display.max_columns', None)

### 
headers = {
    "content-type": "application/json"
}

css = 'body, html, p, h1, .st-emotion-cache-1104ytp h1, [class*="css"] {font-family: "Inter", sans-serif;}'
st.markdown( f'<style>{css}</style>' , unsafe_allow_html= True)


### ---------------------------------------- FUNCTIONS

def frame_name_clean(df):
    
    df = df.rename(columns = {'CountNumber': 'CountNum'})
       
    update_party_names = pd.read_csv('https://raw.githubusercontent.com/jckkrr/Polling-Place-Preference-Tracker---2022-Australian-Election/refs/heads/main/update_party_names.csv').to_dict()
    update_party_names = dict(zip(update_party_names['OldNm'].values(), update_party_names['NewNm'].values()))
        
    if 'PartyNm' in df.columns:
        for party_name in update_party_names.keys():
            df.PartyNm = np.where(df.PartyNm == party_name, update_party_names[party_name], df.PartyNm)
    df['PartyNm'] = np.where(df['PartyNm'] == 'Independent', ' ' + df['GivenNm'] + ' ' + df['Surname'] + ' (IND)', df['PartyNm'])
        
    update_party_abbreviations = pd.read_csv('https://raw.githubusercontent.com/jckkrr/Polling-Place-Preference-Tracker---2022-Australian-Election/refs/heads/main/update_party_abbreviations.csv').to_dict()
    update_party_abbreviations = dict(zip(update_party_abbreviations['OldAb'].values(), update_party_abbreviations['NewAb'].values()))
    if 'PartyAb' in df.columns:
        for party_ab in update_party_abbreviations.keys():
            df.PartyAb = np.where(df.PartyAb == party_ab, update_party_abbreviations[party_ab], df.PartyAb)
    df['PartyAb'] = np.where(df['PartyAb'] == 'IND', 'IND' + '_' + df['Surname'].str.replace(' ','').str[0:4], df['PartyAb'])

    return df

def select_census_data_MF(g):
        
    df_mf = pd.read_csv(f'https://raw.githubusercontent.com/jckkrr/Polling-Place-Preference-Tracker---2022-Australian-Election/refs/heads/main/data/2021Census_{g}_AUST_CED.csv')    

    col_stubs = list(set([x[:-2] for x in df_mf.columns if x not in ['CED_CODE_2021']]))
    for stub in col_stubs:
        df_mf[stub + '_proportion_men'] = df_mf[stub + '_M'] / df_mf[stub + '_P']

    df_mf = df_mf[['CED_CODE_2021'] + [x for x in df_mf.columns if x.endswith('_proportion_men')]]

    return df_mf
    
def select_census_data_allcols(g):
    df_x = pd.read_csv(f'https://raw.githubusercontent.com/jckkrr/Polling-Place-Preference-Tracker---2022-Australian-Election/refs/heads/main/data/2021Census_{g}_AUST_CED.csv')  
    return df_x
    

### _________________________________________ RUN

st.markdown("**Open Investigation Tools** | [constituent.online](%s)" % 'http://www.constituent.online')
    
st.title('Parli Scatterplotter - 2022 Australian Election')
st.write("Explore that factors that might have influenced a party\'s vote.")


### 

df_MAIN = pd.read_csv('https://raw.githubusercontent.com/jckkrr/Polling-Place-Preference-Tracker---2022-Australian-Election/refs/heads/main/data/2022%20Australian%20Election%20AEC%20Data%20-%20HouseDopByDivisionDownload-27966.csv', skiprows = 0, header = 1)
df_MAIN = frame_name_clean(df_MAIN)

## 

df_turnout = pd.read_csv('https://raw.githubusercontent.com/jckkrr/Polling-Place-Preference-Tracker---2022-Australian-Election/refs/heads/main/data/2022%20Australian%20Election%20AEC%20Data%20-%20HouseTurnoutByDivisionDownload-27966.csv', skiprows=[0])
df_informal = pd.read_csv('https://raw.githubusercontent.com/jckkrr/Polling-Place-Preference-Tracker---2022-Australian-Election/refs/heads/main/data/2022%20Australian%20Election%20AEC%20Data%20-%20HouseInformalByDivisionDownload-27966.csv', skiprows=[0])
df_votetypes = pd.read_csv('https://raw.githubusercontent.com/jckkrr/Polling-Place-Preference-Tracker---2022-Australian-Election/refs/heads/main/data/2022%20Australian%20Election%20AEC%20Data%20-%20HouseVotesCountedByDivisionDownload-27966.csv', skiprows=[0])
df_votetypes = df_votetypes[[c for c in df_votetypes.columns if c not in ['Enrolment', 'TotalVotes']]]
df_ADDITIONAL = df_turnout.merge(df_informal, on = ['DivisionID', 'DivisionNm', 'StateAb'])
df_ADDITIONAL = df_ADDITIONAL.merge(df_votetypes, on = ['DivisionID', 'DivisionNm', 'StateAb'])

###

select_census_data_allcols('G02')
    
df_g01 = select_census_data_MF('G01')
df_g02 = select_census_data_allcols('G02')

df_CENSUS = df_g01.merge(df_g02, on = 'CED_CODE_2021', how = 'outer')
df_CENSUS.insert(0, "DivisionID", df_CENSUS['CED_CODE_2021'].str.replace('CED','').astype(int))
df_CENSUS = df_MAIN[['DivisionID', 'DivisionNm', 'StateAb']].copy().drop_duplicates().merge(df_CENSUS, on = 'DivisionID')
df_CENSUS.sample(3)

###

df_YOPTIONS = df_ADDITIONAL.merge(df_CENSUS, on = ['DivisionID', 'DivisionNm', 'StateAb'], how = 'outer')


###


col1, col2 = st.columns([2,3])
with col1: 
    parties = set([x if '(IND)' not in x else 'IND' for x in df_MAIN['PartyNm'].unique() if pd.notnull(x)])
    first_parties = ['Labor', 'Liberal', 'The Nationals', 'The Greens', 'Pauline Hanson\'s One Nation', 'United Australia Party', 'Country Liberal Party',]
    parties = first_parties + sorted([x for x in parties if x not in first_parties])
    chosen_party = st.selectbox('Party:', (parties))
    #chosen_party =  df_MAIN[df_MAIN['PartyNm'] == chosen_party, 'PartyAb'].values[0]

with col2: 
    y_cols = [x for x in df_YOPTIONS.columns if x not in ['DivisionID', 'DivisionNm', 'StateAb', 'CED_CODE_2021']]
    y_cols = sorted(y_cols)
    chosen_ycol = st.selectbox('Y COL:', (y_cols))




###

def make_scatter_firstpref_vs_(x_col_party, y_col, marker_color):

    df_x = df_MAIN.loc[(df_MAIN['PartyNm'] == x_col_party) & (df_MAIN['CountNum'] == 0) & (df_MAIN['CalculationType'] == 'Preference Percent')][['DivisionNm', 'PartyNm', 'CountNum', 'CalculationType', 'CalculationValue']]
    df_plot = df_x.merge(df_YOPTIONS, on = 'DivisionNm', how = 'outer')
    df_plot = df_plot[['DivisionNm', 'StateAb', 'CalculationValue', y_col]].dropna()      

    correlation = round(df_plot[['CalculationValue', y_col]].corr().values[1,0], 4)
    correlation_strength = 'None' if abs(correlation) < 0.15 else 'Very Weak' if abs(correlation) < 0.3 else 'Weak' if abs(correlation) < 0.5 else 'Moderate' if abs(correlation) < 0.7 else 'Strong'
                
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            mode = 'markers',

            x = df_plot['CalculationValue'],
            y = df_plot[y_col],

            marker = dict(size = 10, color = marker_color, symbol = 'circle-open'),

            customdata=df_plot[['DivisionNm', 'StateAb', 'CalculationValue', y_col]],
            hovertemplate= '<span style = "font-family: Inter">'
                "<b>%{customdata[0]} (%{customdata[1]})</b><br>" +
                x_col_party + " FP%: %{x}% <br>" +
                y_col.replace("Percentage","").replace("Percent","") + ": %{y}% <br>" +
                #"%{customdata[1]} \t votes to UAPP<br>" +
                "<extra></extra></span>",

        )
    )

    
    ### ADD TREND LINE
    slope, intercept, r, p, std_err = stats.linregress(df_plot['CalculationValue'], df_plot[y_col])
    def myfunc(x):
      return slope * x + intercept
    model_data = list(map(myfunc, df_plot['CalculationValue']))
    fig.add_trace(
        go.Scatter(
            mode = 'lines',
            x=df_plot['CalculationValue'],
            y=model_data, 
            line_shape='linear',
            line = dict(color = 'rgba(0,200,22,0.33)')
        )
    )
    
    fig.update_layout(title = f'<b><span style = "font-family: Inter">{x_col_party}</b> <span style = "font-weight: normal"> {y_col} <span style = "color: silver"> Correlation: {correlation_strength} ({correlation})</span></span><br><sup><span style = "font-weight: normal">2022 Australia election</span></span></sup>')
    customChartDefaultStyling.styling(fig)
    fig.update_xaxes(title = f'<b><span style = "font-size: 10px">{x_col_party} first preferences (%)</span></b>',)
    fig.update_yaxes(title = f'<b><span style = "font-size: 10px; font-family: Inter">{ y_col.replace("Percentage"," (%)").replace("Percent"," (%)")}</span></b>',)
    fig.update_layout(width = 500, height = 500,showlegend = False)

    
    st.plotly_chart(fig, use_container_width=True)

    
make_scatter_firstpref_vs_(chosen_party, chosen_ycol, '#181818')
