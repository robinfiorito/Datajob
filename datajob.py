import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_1 = pd.read_csv("kaggle_survey_2020_responses.csv", header = 1)
df_2 = pd.read_csv("kaggle_survey_2020_responses.csv", header = 0)
df_2 = df_2.iloc[1:]
df= pd.read_csv("kaggle_survey_2020_responses.csv", header = 0)
df= df.iloc[1:]
st.title("Projet de rapport : Data job")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization","Preprocessing des données","Modélisation","Optimisation","Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
cols_elimin = ["Time from Start to Finish (seconds)","Q1","Q2","Q3","Q4","Q8","Q7_Part_12","Q7_OTHER","Q9_Part_11","Q9_OTHER","Q10_Part_13",
               "Q10_OTHER","Q11","Q12_Part_3","Q12_OTHER","Q13","Q14_Part_11","Q14_OTHER","Q16_Part_15","Q16_OTHER","Q17_Part_11",
               "Q17_OTHER","Q18_Part_6","Q18_OTHER","Q19_Part_5","Q19_OTHER","Q23_Part_7","Q23_OTHER","Q20","Q21","Q22",
               "Q26_A_Part_11","Q26_A_OTHER","Q24","Q25","Q27_A_Part_11","Q27_A_OTHER","Q28_A_Part_10","Q28_A_OTHER",
               "Q29_A_Part_17","Q29_A_OTHER","Q30","Q32","Q31_A_Part_14","Q31_A_OTHER","Q33_A_Part_7","Q33_A_OTHER",
               "Q34_A_Part_11","Q34_A_OTHER","Q35_A_Part_10","Q35_A_OTHER","Q36_Part_9","Q36_OTHER",
               "Q37_Part_11","Q37_OTHER","Q39_Part_11","Q39_OTHER",
               "Q26_B_Part_1","Q26_B_Part_2","Q26_B_Part_3","Q26_B_Part_4","Q26_B_Part_5","Q26_B_Part_6",
              "Q26_B_Part_7","Q26_B_Part_8","Q26_B_Part_9","Q26_B_Part_10","Q26_B_Part_11","Q26_B_OTHER",
               "Q27_B_Part_1","Q27_B_Part_2","Q27_B_Part_3","Q27_B_Part_4","Q27_B_Part_5","Q27_B_Part_6",
              "Q27_B_Part_7","Q27_B_Part_8","Q27_B_Part_9","Q27_B_Part_10","Q27_B_Part_11","Q27_B_OTHER",
               "Q28_B_Part_1","Q28_B_Part_2","Q28_B_Part_3","Q28_B_Part_4","Q28_B_Part_5","Q28_B_Part_6",
              "Q28_B_Part_7","Q28_B_Part_8","Q28_B_Part_9","Q28_B_Part_10","Q28_B_OTHER",
               "Q29_B_Part_1","Q29_B_Part_2","Q29_B_Part_3","Q29_B_Part_4","Q29_B_Part_5","Q29_B_Part_6",
              "Q29_B_Part_7","Q29_B_Part_8","Q29_B_Part_9","Q29_B_Part_10","Q29_B_Part_11","Q29_B_Part_12",
              "Q29_B_Part_13","Q29_B_Part_14","Q29_B_Part_15","Q29_B_Part_16","Q29_B_Part_17","Q29_B_OTHER",
               "Q31_B_Part_1","Q31_B_Part_2","Q31_B_Part_3","Q31_B_Part_4","Q31_B_Part_5","Q31_B_Part_6",
              "Q31_B_Part_7","Q31_B_Part_8","Q31_B_Part_9","Q31_B_Part_10","Q31_B_Part_11","Q31_B_Part_12",
              "Q31_B_Part_13","Q31_B_Part_14","Q31_B_OTHER","Q33_B_Part_1","Q33_B_Part_2","Q33_B_Part_3",
               "Q33_B_Part_4","Q33_B_Part_5","Q33_B_Part_6","Q33_B_Part_7","Q33_B_OTHER",
               "Q34_B_Part_1","Q34_B_Part_2","Q34_B_Part_3","Q34_B_Part_4","Q34_B_Part_5","Q34_B_Part_6",
              "Q34_B_Part_7","Q34_B_Part_8","Q34_B_Part_9","Q34_B_Part_10","Q34_B_Part_11","Q34_B_OTHER",
               "Q35_B_Part_1","Q35_B_Part_2","Q35_B_Part_3","Q35_B_Part_4","Q35_B_Part_5","Q35_B_Part_6",
              "Q35_B_Part_7","Q35_B_Part_8","Q35_B_Part_9","Q35_B_Part_10","Q35_B_OTHER","Q36_Part_1","Q36_Part_2","Q36_Part_3","Q36_Part_4",
                "Q36_Part_5","Q36_Part_6","Q36_Part_7",
            "Q36_Part_8","Q37_Part_1","Q37_Part_2","Q37_Part_3","Q37_Part_4","Q37_Part_5","Q37_Part_6","Q37_Part_7",
            "Q37_Part_8","Q37_Part_9","Q37_Part_10","Q39_Part_1","Q39_Part_2","Q39_Part_3","Q39_Part_4","Q39_Part_5",
                "Q39_Part_6","Q39_Part_7","Q39_Part_8","Q39_Part_9","Q39_Part_10"]
df_clean = df.drop(cols_elimin, axis = 1)
df_clean = df_clean.loc[(df_clean["Q5"] != "Student") & (df_clean["Q5"] != "Other") & (df_clean["Q5"] != "Currently not employed")
             & (df_clean["Q5"] != "Product/Project Manager")& (df_clean["Q5"] != "Business Analyst")
             & (df_clean["Q5"] != "Research Scientist")]
df_clean = df_clean.dropna(subset=['Q5'])
df_clean = df_clean.loc[df_clean["Q6"] != "I have never written code"]
df_clean["Q6"] = df_clean["Q6"].replace("< 1 years","0-5 years")
df_clean["Q6"] = df_clean["Q6"].replace("1-2 years","0-5 years")
df_clean["Q6"] = df_clean["Q6"].replace("3-5 years","0-5 years")
df_clean["Q6"] = df_clean["Q6"].replace("10-20 years","10+ years")
df_clean["Q6"] = df_clean["Q6"].replace("20+ years","10+ years")
df_clean["Q15"] = df_clean["Q15"].replace("Under 1 year","0-5 years")
df_clean["Q15"] = df_clean["Q15"].replace("1-2 years","0-5 years")
df_clean["Q15"] = df_clean["Q15"].replace("2-3 years","0-5 years")
df_clean["Q15"] = df_clean["Q15"].replace("3-4 years","0-5 years")
df_clean["Q15"] = df_clean["Q15"].replace("4-5 years","0-5 years")
df_clean["Q15"] = df_clean["Q15"].replace("10-20 years","10+ years")
df_clean["Q15"] = df_clean["Q15"].replace("20 or more years","10+ years")
col_binaires = ["Q7_Part_1","Q7_Part_2","Q7_Part_3","Q7_Part_4","Q7_Part_5","Q7_Part_6","Q7_Part_7","Q7_Part_8",
                "Q7_Part_9","Q7_Part_10","Q7_Part_11","Q9_Part_1","Q9_Part_2","Q9_Part_3","Q9_Part_4","Q9_Part_5",
                "Q9_Part_6","Q9_Part_7","Q9_Part_8","Q9_Part_9","Q9_Part_10","Q10_Part_1","Q10_Part_2","Q10_Part_3",
                "Q10_Part_4","Q10_Part_5","Q10_Part_6","Q10_Part_7","Q10_Part_8","Q10_Part_9","Q10_Part_10",
                "Q10_Part_11","Q10_Part_12","Q12_Part_1","Q12_Part_2","Q14_Part_1","Q14_Part_2","Q14_Part_3",
                "Q14_Part_4","Q14_Part_5","Q14_Part_6","Q14_Part_7","Q14_Part_8","Q14_Part_9","Q14_Part_10",
                "Q16_Part_1","Q16_Part_2","Q16_Part_3","Q16_Part_4","Q16_Part_5","Q16_Part_6","Q16_Part_7",
            "Q16_Part_8","Q16_Part_9","Q16_Part_10","Q16_Part_11","Q16_Part_12","Q16_Part_13","Q16_Part_14","Q17_Part_1","Q17_Part_2","Q17_Part_3","Q17_Part_4","Q17_Part_5",
                "Q17_Part_6","Q17_Part_7","Q17_Part_8","Q17_Part_9","Q17_Part_10",
                "Q18_Part_1","Q18_Part_2","Q18_Part_3","Q18_Part_4","Q18_Part_5","Q19_Part_1",
                "Q19_Part_2","Q19_Part_3","Q19_Part_4","Q23_Part_1","Q23_Part_2","Q23_Part_3","Q23_Part_4","Q23_Part_5","Q23_Part_6","Q26_A_Part_1","Q26_A_Part_2","Q26_A_Part_3","Q26_A_Part_4","Q26_A_Part_5","Q26_A_Part_6",
              "Q26_A_Part_7","Q26_A_Part_8","Q26_A_Part_9","Q26_A_Part_10","Q27_A_Part_1","Q27_A_Part_2",
                "Q27_A_Part_3","Q27_A_Part_4","Q27_A_Part_5","Q27_A_Part_6",
              "Q27_A_Part_7","Q27_A_Part_8","Q27_A_Part_9","Q27_A_Part_10","Q28_A_Part_1","Q28_A_Part_2",
                "Q28_A_Part_3","Q28_A_Part_4","Q28_A_Part_5","Q28_A_Part_6",
              "Q28_A_Part_7","Q28_A_Part_8","Q28_A_Part_9","Q29_A_Part_1","Q29_A_Part_2","Q29_A_Part_3",
                "Q29_A_Part_4","Q29_A_Part_5","Q29_A_Part_6",
              "Q29_A_Part_7","Q29_A_Part_8","Q29_A_Part_9","Q29_A_Part_10","Q29_A_Part_11","Q29_A_Part_12",
              "Q29_A_Part_13","Q29_A_Part_14","Q29_A_Part_15","Q29_A_Part_16","Q31_A_Part_1","Q31_A_Part_2",
                "Q31_A_Part_3","Q31_A_Part_4","Q31_A_Part_5","Q31_A_Part_6",
              "Q31_A_Part_7","Q31_A_Part_8","Q31_A_Part_9","Q31_A_Part_10","Q31_A_Part_11","Q31_A_Part_12",
              "Q31_A_Part_13","Q33_A_Part_1","Q33_A_Part_2","Q33_A_Part_3","Q33_A_Part_4","Q33_A_Part_5","Q33_A_Part_6",
            "Q34_A_Part_1","Q34_A_Part_2","Q34_A_Part_3","Q34_A_Part_4","Q34_A_Part_5","Q34_A_Part_6",
              "Q34_A_Part_7","Q34_A_Part_8","Q34_A_Part_9","Q34_A_Part_10","Q35_A_Part_1","Q35_A_Part_2","Q35_A_Part_3",
                "Q35_A_Part_4","Q35_A_Part_5","Q35_A_Part_6",
              "Q35_A_Part_7","Q35_A_Part_8","Q35_A_Part_9"]
X_binaires = df_clean[col_binaires]
X_cat = df_clean.drop(col_binaires, axis = 1)
y = X_cat["Q5"]
X_cat = X_cat.drop("Q5", axis = 1)
ohe = OneHotEncoder(drop="first", sparse_output=False)
oe = OrdinalEncoder(categories = [["0-5 years",
                                       "5-10 years","10+ years","Missing"],["I do not use machine learning methods",
                                       "0-5 years","5-10 years","10+ years","Missing"]])
X_ordinal = X_cat[["Q6","Q15"]]
X_ordinal = X_ordinal.fillna("Missing")
X_encoded = oe.fit_transform(X_ordinal)
X_ordinal = pd.DataFrame(X_encoded, columns = X_ordinal.columns)
X_ordinal["Q6"] = X_ordinal["Q6"] + 1
X_ordinal = X_ordinal.replace(4,np.nan)

X_OneHot = X_cat.drop(["Q6","Q15"], axis = 1)
X_encoded = ohe.fit_transform(X_OneHot)
X_encoded = pd.DataFrame(X_encoded, columns=ohe.get_feature_names_out(['Q38']))
def renommer_colonnes(df_clean):
     nouveaux_noms = {}
     for col in df_clean.columns:
        valeur_unique = df_clean[col].dropna().unique()
        if len(valeur_unique) == 1:
           nouveaux_noms[col] = (col[:3] + str(valeur_unique[0]))
        else :
           nouveaux_noms[col] = col
     return df_clean.rename(columns=nouveaux_noms)
X_binaires = renommer_colonnes(X_binaires)
X_binaires = X_binaires.apply(lambda x: np.where(pd.isna(x), np.nan, 1))
X_binaires = X_binaires.replace(np.nan,0)
X_binaires = X_binaires.reset_index()
X_binaires = X_binaires.drop("index", axis = 1)
X_binaires = pd.concat([X_binaires,X_encoded,X_ordinal], axis = 1)
X_binaires = X_binaires.drop(["Q38_Other","Q38_nan"], axis = 1)
X_binaires["Q6"] = X_binaires["Q6"].replace(1,0.3333)
X_binaires["Q6"] = X_binaires["Q6"].replace(2,0.6667)
X_binaires["Q6"] = X_binaires["Q6"].replace(3,1)
X_binaires["Q15"] = X_binaires["Q15"].replace(1,0.3333)
X_binaires["Q15"] = X_binaires["Q15"].replace(2,0.6667)
X_binaires["Q15"] = X_binaires["Q15"].replace(3,1)
   # Ne pas itérer deux fois cette cellule
X_scaled = X_binaires
imputer = KNNImputer(n_neighbors=5)
X_scaled = pd.DataFrame(imputer.fit_transform(X_scaled), columns = X_scaled.columns)
variance = X_scaled.var()
   # Filtrer les colonnes avec une variance inférieure à 0.06 (seuil arbitraire)
columns_to_remove = variance[variance < 0.06].index
X_scaled = X_scaled.drop(columns=columns_to_remove)
le = LabelEncoder()
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.2, random_state = 42)
y_labels = np.unique(y_train)
y_test_labels = np.unique(y_test)
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

if page == pages[0] : 
  st.write("### Introduction")
  st.write("Le dataset présenté ici est issu d'une enquete sur les métiers de la Data Industry menée par le site kaggle.com .")
  st.write("Le dataset présente ", df_1.shape[0], " lignes et ",df_1.shape[1], " colonnes."    )
  if st.checkbox("Afficher les colonnes du Dataset") :
     st.dataframe(df_1.columns)
  if st.checkbox("Afficher un aperçu des réponses") :
     st.dataframe(df_1.head(5))


if page == pages[1] : 
  
  st.write("### DataVizualization")
  if st.checkbox("Distribution des données") :
      fig = plt.figure()
      Q2 = df_2["Q2"].value_counts(normalize=True).values 
      Q2_liste = df_2["Q2"].value_counts().index.tolist()  
      plt.bar(Q2_liste,Q2,color = "r")
      plt.xticks(Q2_liste,["Man","Woman","Non defined","Self described","Non binary"])
      plt.title("Répartition par genre")
      st.pyplot(fig)

      Q1 = df_2["Q1"].value_counts(normalize=True).sort_index().values
      Q1_liste = df_2["Q1"].value_counts().sort_index().index.tolist()
      fig_2 = plt.figure()
      plt.bar(Q1_liste,
       Q1
       )
      plt.title("Age de la population étudiée")
      st.pyplot(fig_2)
  
  if st.checkbox("Rôles professionnels") :
      fig = plt.figure(figsize = (15, 5))
      df_graph = df_2[df_2["Q2"].isin(["Man","Woman"])]
      ax2 = sns.countplot(x = "Q5", hue = "Q2", data = df_graph)
      ax2.set(xlabel = "Métier", ylabel = "Effectif")
      ax2.legend(title = "Genre")
      plt.title("Répartition du genre des participants par métier")
      plt.xticks(rotation = 45, fontsize = 8)
      st.pyplot(fig)

  if st.checkbox("Origine géographique") :  
      Q3 = df_2["Q3"].value_counts(normalize=True).values
      Q3_liste = df_2["Q3"].value_counts().index.tolist()
      for i in range(len(Q3_liste)):
        if Q3_liste[i] == "United Kingdom of Great Britain and Northern Ireland":
           Q3_liste[i] = "UK"
        if Q3_liste[i] == "United States of America":
           Q3_liste[i] = "USA"
      df_Q3 = pd.DataFrame({
      'Réponses': Q3_liste,
      'Pourcentages': Q3
      })
      df_Q3 = df_Q3.head(10)
      Q3_other = Q3[10:].sum()
      df_Q3.loc[df_Q3["Réponses"] == "Other","Pourcentages"] += Q3_other

      Q3_explode = []
      i=0
      while i < len(df_Q3["Réponses"]):
        i += 1
        Q3_explode.append(0.03)


      fig = plt.figure()
      plt.pie(x = df_Q3["Pourcentages"],
        labels = df_Q3["Réponses"],
        autopct = lambda x : str(round(x,1))+" %",
        explode = Q3_explode,
        labeldistance = 1.05,
        pctdistance = 1.45,
        wedgeprops={'linewidth': 2.0, 'edgecolor': 'white'},
        textprops={'size': 'x-small'}
        )
      plt.title("Répartition des pays d'origine", fontsize = 10, pad = 50)
      st.pyplot(fig)

  if st.checkbox("Niveau d'études") :  
        df_2 = df_2.sort_values(by=["Q6"])
        Q6 = df_2["Q6"].value_counts(normalize=True).values
        Q6_liste = df_2["Q6"].value_counts().index.tolist()
        df_Q6 = pd.DataFrame({
         'Réponses': Q6_liste,
         'Pourcentages': Q6
        })
        order_Q6 = {
        'I have never written code': 1,
        '< 1 years': 2,
        '1-2 years': 3,
        '3-5 years': 4,
        '5-10 years': 5,
        '10-20 years': 6,
        '20+ years' : 7
        }
  
        df_Q6["ordre"] = df_Q6["Réponses"].map(order_Q6)
        df_Q6 = df_Q6.sort_values(by = "ordre").drop(columns = ["ordre"])
        fig = plt.figure()
        plt.barh(df_Q6["Réponses"],
        df_Q6["Pourcentages"],
        color = "y"
        )
        plt.title("Année de pratique de codage")
        st.pyplot(fig)

        Q4 = df["Q4"].value_counts(normalize=True).values
        Q4_liste = df["Q4"].value_counts().index.tolist()
        Q4_explode = []
        i=0
        while i < len(Q4):
           i += 1
           Q4_explode.append(0.1)
        fig_2 = plt.figure()
        plt.pie(x = Q4,
            labels = Q4_liste,
            autopct = lambda x : str(round(x,1))+" %",
            explode = Q4_explode,
            labeldistance = 1.2,
            pctdistance = 1.4,
            wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'},
            textprops={'size': 'x-small'}
            )
        plt.title("Répartition par niveau d'études", fontsize = 10, pad = 50)
        st.pyplot(fig_2)

  if st.checkbox("Tendances d'utilisation de la plateforme") :      
        Q7 = ["Q7_Part_1", "Q7_Part_2", "Q7_Part_3", "Q7_Part_4", "Q7_Part_5", "Q7_Part_6", "Q7_Part_7", "Q7_Part_8", "Q7_Part_9",
        "Q7_Part_10", "Q7_Part_11", "Q7_Part_12", "Q7_OTHER"]
        Q5_Q7 = df[["Q5"] + Q7]
        Q5_Q7 = Q5_Q7[Q5_Q7["Q5"] != "Student"]
        Data_1 = pd.melt(Q5_Q7, id_vars=["Q5"], value_vars=Q7, var_name="Q7", value_name="Programming Language")
        Data_1 = Data_1.dropna()
        Data_1_grp = Data_1.groupby(["Q5", "Q7"])["Programming Language"].count().reset_index()
        fig_1 = plt.figure()
        px.bar(Data_1_grp, x = "Programming Language", y = "Q7", color = "Q5",
                title = "Programming Language utilisés régulièrement selon l'occupation",
                labels = {"Q7": "Programming Language", "Q5": "Occupation"}
                )
        y_labels_1 = ["Python", "R", "SQL", "C", "C++", "Java", "Javascript", "Julia", "Swift", "Bash", "Matlab", "None", "Other"]
        #fig_1.update_layout(xaxis_tickangle = -45, xaxis_title = None, legend_title_text = "Occupation")
        #fig_1.update_yaxes(ticktext = y_labels_1, tickvals = Q7)
        st.pyplot(fig_1)

if page == pages[2] : 
   st.write("### Preprocessing des données")
   st.write("#### 1) Cleaning des données")
   st.write("Avant de procéder à la modélisation des données, il a fallu procédér au cleaning des données :")
   st.markdown(" - Élimination des colonnes non pertinentes.")
   st.markdown(" - Élimination des réponses de participants ne travaillant pas dans la data.")
   st.markdown(" - Élimination des répondants n'ayant jamais écrit de code.")
   st.markdown(" - Suppression des colonnes déclaratives et non pertinentes.")
   st.write("#### 2) Encodage et transformation")
   st.write("Une deuxième série de transformations a été nécessaire pour rendre les données exploitables par les modèles de machine learning :")   
   st.markdown(" - Ordinal Encoding des variables avec des catégories ordonnées.")
   st.markdown(" - Label Encoding de la valeur cible.")
   st.markdown(" - One Hot Encoding des variables catégorielles sans ordre défini.")
   st.write("#### 3) Imputation des valeurs manquantes")
   st.write("Une troisième série de transformations pour remplacer les valeurs nulles.")   
   st.markdown(" - La méthode d'imputation par KNN a été choisie pour remplacer les valeurs manquantes")
   st.write("#### 4) Analyse de la variance et corrélation des variables")
   st.write("La dernière série de transformations est orientée à améliorer la performance du modèle :")   
   st.markdown(" - Analyse de la variance des variables. Aucune variable présentait une variance trop faible.")   
   st.markdown(" - Analyse de la corrélation des variables. Aucune variable n'a été éliminée.")   
 
if page == pages[4] :

   from sklearn.metrics import f1_score
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.svm import SVC
   from sklearn.linear_model import LogisticRegression
   import xgboost as xgb
   from xgboost import XGBClassifier
   st.write("### Modélisation")
   st.write(" Les optimisations testées sont les suivantes :")
   st.markdown("- PCA")
   st.markdown("- SMOTE")
   st.markdown("- RandomOverSampler")
   st.markdown("- RandomUnderSampler")
   st.markdown("- ClusterCentroids")
   st.markdown("- GridSearch")
   st.markdown("- RandomizedSearchCV")

   def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier(max_depth = 8, min_samples_split=10, min_samples_leaf=10, n_estimators=150, random_state=42)
    elif classifier == "XGBoost" :
        clf = XGBClassifier(max_depth=3,learning_rate=0.1, n_estimators=100,subsample=0.8, colsample_bytree=0.8, random_state = 42)
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression(multi_class="multinomial", solver="saga", penalty= "l2", max_iter= 100, C= 0.1,random_state=42)
    clf.fit(X_train, y_train) 
    return clf

   def scores(clf, choice):
    y_pred = clf.predict(X_test)
    if choice == 'Test score':
        return clf.score(X_test, y_test)
    elif choice == 'f1 score':
        f1 = f1_score(y_test,y_pred, average=None)
        return pd.DataFrame({"Métier": y_labels, "f1 score": f1})
    elif choice == "Train score":
        return clf.score(X_train, y_train)
    
   choix = ['Random Forest', 'Logistic Regression',"XGBoost"]
   option = st.selectbox('Choix du modèle', choix)
   st.write('Le modèle choisi est :', option)
   clf = prediction(option)
   display = st.radio('Que souhaitez-vous montrer ?', ('f1 score', 'Train score', "Test score"))
   st.write(scores(clf, display))

if page == pages[3] :

   from sklearn.metrics import f1_score
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.svm import SVC
   from sklearn.linear_model import LogisticRegression
   import xgboost as xgb
   from xgboost import XGBClassifier

   st.write("### Modélisation")
   st.write(" 6 différents modèles ont été testés :")
   st.markdown("- Decision Tree Classifier")
   st.markdown("- XGBoost Classifier")
   st.markdown("- Support Vector Classifier")
   st.markdown("- AdaBoost sur le modèle Decision Tree Classifier")
   st.markdown("- Logistic Regression")
   st.markdown("- Random Forest Classifier")

   def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier(random_state=42)
    elif classifier == "XGBoost" :
        clf = XGBClassifier(random_state=42)
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train) 
    return clf

   def scores(clf, choice):
    y_pred = clf.predict(X_test)
    if choice == 'Test score':
        return clf.score(X_test, y_test)
    elif choice == 'f1 score':
        return f1_score(y_test,y_pred, average=None)
    elif choice == "Train score":
        return clf.score(X_train, y_train)
    
   choix = ['Random Forest', 'Logistic Regression',"XGBoost"]
   option = st.selectbox('Choix du modèle', choix)
   st.write('Le modèle choisi est :', option)
   clf = prediction(option)
   display = st.radio('Que souhaitez-vous montrer ?', ('f1 score', 'Train score', "Test score"))
   st.write(scores(clf, display))

if page == pages[5]:
   st.write("### Conclusions et perspectives")
   st.write("#### 1) Conclusions : ")
   st.markdown(" - Jeu de données en accord avec la réalité")
   st.markdown(" - Performance des modèle faible : max 55%")
   st.markdown(" - Limitation du surapprentissage")
   st.write("#### 2) Pistes à envisager pour améliorer la performance des modèles : ")   
   st.markdown(" - Améliorarion de la qualité des données")
   st.markdown(" - Exploration d'autres algorithmes")
   st.markdown(" - Affinement des stratégies d'échantillonnage")