import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


df_1 = pd.read_csv("kaggle_survey_2020_responses.csv", header = 1)
df_2 = pd.read_csv("kaggle_survey_2020_responses.csv", header = 0)
df_2 = df_2.iloc[1:]
df= pd.read_csv("kaggle_survey_2020_responses.csv", header = 0)
df= df.iloc[1:]
st.title("Projet de rapport _ Data job")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

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
  fig = plt.figure()
  Q2 = df_2["Q2"].value_counts(normalize=True).values 
  Q2_liste = df_2["Q2"].value_counts().index.tolist()  
  plt.bar(Q2_liste,Q2,color = "r")
  plt.xticks(Q2_liste,["Man","Woman","Non defined","Self described","Non binary"])
  plt.title("Répartition par genre")
  st.pyplot(fig)
  st.write("La population est constituée à près de 60 % d'hommes.")

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
  st.write("Près de 55% des personnes interrogées ont pratiqué moins de 5 ans de pratique.")

  Q1 = df_2["Q1"].value_counts(normalize=True).sort_index().values
  Q1_liste = df_2["Q1"].value_counts().sort_index().index.tolist()
  fig = plt.figure()
  plt.bar(Q1_liste,
      Q1
       )
  plt.title("Age de la population étudiée")
  st.pyplot(fig)
  st.write("La répartition parage est plus hétérogène. La tranche d'age la plus répandu est entre 25 et 29 ans, avec 20% des répondants. 50% de la population a moins de 30 ans.")

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
  st.write("La population étudiée vient principalement de deux pays, Inde (50%) et USA (20%).")

if page == pages[2] : 
   st.write("### Modélisation")
   df = pd.read_csv("kaggle_survey_2020_responses.csv", header = 0)
   df = df.iloc[1:]
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
   df = df.drop(cols_elimin, axis = 1)
   df = df.loc[(df["Q5"] != "Student") & (df["Q5"] != "Other") & (df["Q5"] != "Currently not employed")
             & (df["Q5"] != "Product/Project Manager")& (df["Q5"] != "Business Analyst")
             & (df["Q5"] != "Research Scientist")]
   df = df.dropna(subset=['Q5'])
   df = df.loc[df["Q6"] != "I have never written code"]
   df["Q6"] = df["Q6"].replace("< 1 years","0-5 years")
   df["Q6"] = df["Q6"].replace("1-2 years","0-5 years")
   df["Q6"] = df["Q6"].replace("3-5 years","0-5 years")
   df["Q6"] = df["Q6"].replace("10-20 years","10+ years")
   df["Q6"] = df["Q6"].replace("20+ years","10+ years")
   df["Q15"] = df["Q15"].replace("Under 1 year","0-5 years")
   df["Q15"] = df["Q15"].replace("1-2 years","0-5 years")
   df["Q15"] = df["Q15"].replace("2-3 years","0-5 years")
   df["Q15"] = df["Q15"].replace("3-4 years","0-5 years")
   df["Q15"] = df["Q15"].replace("4-5 years","0-5 years")
   df["Q15"] = df["Q15"].replace("10-20 years","10+ years")
   df["Q15"] = df["Q15"].replace("20 or more years","10+ years")
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
   X_binaires = df[col_binaires]
   X_cat = df.drop(col_binaires, axis = 1)
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
   def renommer_colonnes(df):
     nouveaux_noms = {}
     for col in df.columns:
        valeur_unique = df[col].dropna().unique()
        if len(valeur_unique) == 1:
           nouveaux_noms[col] = (col[:3] + str(valeur_unique[0]))
        else :
           nouveaux_noms[col] = col
     return df.rename(columns=nouveaux_noms)
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
   X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.2,random_state = 32)
   y_train = le.fit_transform(y_train)
   y_test = le.transform(y_test)

   def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestClassifier()
    elif classifier == 'SVC':
        clf = SVC()
    elif classifier == 'Logistic Regression':
        clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf
   def scores(clf, choice):
    if choice == 'Accuracy':
        return clf.score(X_test, y_test)
    elif choice == 'Confusion matrix':
        return confusion_matrix(y_test, clf.predict(X_test))
   choix = ['Random Forest', 'SVC', 'Logistic Regression']
   option = st.selectbox('Choix du modèle', choix)
   st.write('Le modèle choisi est :', option)
   clf = prediction(option)
   display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
   if display == 'Accuracy':
    st.write(scores(clf, display))
   elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))
