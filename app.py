import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.figure_factory as ff
st.set_option('deprecation.showPyplotGlobalUse', False)



# Import des données
df_model = pd.read_csv('data_model_sampled.csv',index_col = 'SK_ID_CURR')
df_model=df_model.drop(columns=['TARGET','Unnamed: 0'])
# Modification des données en chaîne de charactères pour la transposition
df = df_model.astype(np.dtype(str))

# Chargement du modèle
model = pickle.load(open('RFC_best.sav', 'rb'))

# Instanciation des différentes section
header = st.container()
result_ml = st.container()
info_client = st.container()
info_comp = st.container()
feature_imp = st.container()
filtered_dataset = st.container()

st.sidebar.image('data:image/png;base64,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', width=200)
st.sidebar.image('https://pixabay.com/fr/images/download/brain-6215574_640.jpg', width=200)
# Section 1 de la sidebar
st.sidebar.header('Sélection du numéro client')
id_client = st.sidebar.selectbox('Identifiant client', df_model.index)

# Affichage du titre et des données relatives au client sélectionné
with header :
    st.title('''Tableau de bord d'allocation de crédit client''' )
    #st.write(df_model)

df_client = df.loc[df.index == int(id_client)].transpose()
df_client.columns = ['Informations clients']

with info_client :
    st.header('''Informations personnelles''')
    st.write(df_client)

st.write(df)
# Préparation des graphiques comparatifs
amt_inc_total = np.log(df_model.loc[df_model.index == int(id_client), 'AMT_INCOME_TOTAL'].values[0])
x_a = [np.log(df_model['AMT_INCOME_TOTAL'])]
fig_a = ff.create_distplot(x_a,['AMT_INCOME_TOTAL'], bin_size=0.3)
fig_a.add_vline(x=amt_inc_total, annotation_text=' Vous êtes ici')


x_b = [np.log(df_model['AMT_CREDIT'])]
var = np.log(df_model.loc[df_model.index == int(id_client), 'AMT_CREDIT'].values[0])
fig_b = ff.create_distplot(x_b,['AMT_CREDIT'], bin_size=0.3)
fig_b.add_vline(x=var, annotation_text=' Vous êtes ici')


#x_c = [np.log(df_model['SUM(previous.AMT_APPLICATION)']+1)]
#var = np.log(df_model.loc[df_model.index == int(id_client), 'SUM(previous.AMT_APPLICATION)'].values[0])
#fig_c = ff.create_distplot(x_c,['SUM(previous.AMT_APPLICATION)'], bin_size=0.3)
#fig_c.add_vline(x=var, annotation_text=' Vous êtes ici')

# Visualisation des graphiques
with info_comp :
    st.header('''Informations comparatives''')
    st.subheader('Vos revenus')
    st.plotly_chart(fig_a, use_container_width=True)
    st.subheader('Montant du crédit')
    st.plotly_chart(fig_b, use_container_width=True)
    #st.subheader('Somme des crédits demandés lors des précédentes demandes')
    #st.plotly_chart(fig_c, use_container_width=True)

# Résultat de la modélisation
with result_ml :
    st.header('''Résultat de la demande de crédit''')
    per_pos = model.predict_proba(df_model.loc[df_model.index == int(id_client)])[0][1]
    if per_pos < 0.51229 :
        st.markdown("<p style=color:Green;font-weight:bold> Votre crédit est accepté</p>" , unsafe_allow_html=True)
    else :
        st.markdown("<p style=color:Red;font-weight:bold> Votre crédit est refusé</p>" , unsafe_allow_html=True)
    st.write('Votre crédit est refusé si ce score est supérieur à 0.51229 : {}'.format(round(per_pos,3)))

# Les variables les plus importantes dans la modélisation
with feature_imp :
    st.header('''Variables les plus importantes dans le calcul de l'allocation''')
    var_pos = ['CNT_CHILDREN','CNT_FAM_MEMBERS','AMT_GOODS_PRICE','AMT_CREDIT','FLAG_OWN_CAR', 'DAYS_EMPLOYED', 'DAYS_BIRTH','AMT_ANNUITY']


    var_neg = ['FLAG_PHONE','AMT_INCOME_TOTAL' ,
        ]

    df_feat_imp = df
    st.subheader('Plus ces valeurs sont hautes, plus vous avez de chances que votre crédit soit **accepté ou refusé** (poids les plus grand dans la décision)')
    df_pos = df_feat_imp.loc[df_feat_imp.index == int(id_client)][var_pos].transpose()
    st.write(df_pos)
    #st.subheader('Plus ces valeurs sont hautes, plus vous avez de chances que votre crédit soit **accepté**')
    #df_neg = df_feat_imp.loc[df_feat_imp.index == int(id_client)][var_neg].transpose()
    #st.write(df_neg)

# Outil permettant de sélectionner des variables et visualiser les données par rapport à plusieurs clients
with filtered_dataset :
    st.title('''Explorateur du jeu de données''')
    is_check = st.checkbox("Affichage des données")
    st.sidebar.header('Explorateur du jeu de données')
    columns_display = st.sidebar.multiselect("Étape n°1 : Sélectionner les variables à afficher", df_model.columns)
    columns_display = list(columns_display)

    columns_filter = st.sidebar.multiselect("Étape n°2 : Sélectionner les variables à filtrer", columns_display)
    columns_filter = list(columns_filter)

    min_max = st.checkbox("Ordre croissant")

    if is_check & min_max :
        st.write(df_model[columns_display].sort_values(by=columns_filter, ascending=True))

    elif is_check :
        st.write(df_model[columns_display].sort_values(by=columns_filter, ascending=False))
