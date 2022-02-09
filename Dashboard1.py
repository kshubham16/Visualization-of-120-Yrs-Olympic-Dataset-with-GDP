import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import seaborn as sns
import streamlit as st
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import sklearn.metrics as metrics
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
#from xgboost.xgbclassifier import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

img =st.container()
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with img:
    image = Image.open('ol_img.jpg')
    st.image(image, caption='wallpapercave.com')

with header:
    st.title('120 years of Olympic history: athletes and results!')

st.write("""
### Raw Datasets
""")

athlete_info = st.selectbox(
    " ",
    ("Athlete_Info","Region", "GDP")
)


with dataset:
    def show_data(athlete_info):
        if athlete_info == "Athlete_Info":
            d_set = st.dataframe(pd.read_excel("athlete_events.xlsx"))
        elif athlete_info == "Region":
            d_set = st.dataframe(pd.read_csv("noc_regions.csv"))    
        else:
            d_set = st.dataframe(pd.read_csv("GDP1.csv"))

        # x = data.data
        # # y = data.target
   
        # return x
X = show_data(athlete_info)

st.write("""
### Data Viasualization
""")

#_____________________________________________________________________________________________________________________
#Upload Row Dataset
rw_dataset = pd.read_excel("ath_data.xlsx")
#_____________________________________________________________________________________________________________________

# fig, ax = plt.subplots(figsize=(5, 5))
# st.pyplot(fig, figsize=(5, 5))
st.write("Null Values in our Dataset 📌")
fig, ax = plt.subplots()
sns.heatmap(rw_dataset.isnull(), ax=ax)
st.pyplot(fig)

st.write("Total Participants 👨‍👨‍👧‍👧")
df_gen = pd.DataFrame(rw_dataset['Sex'].value_counts())
st.bar_chart(df_gen)

st.write("Percentage(%) of participants")
fig, ax = plt.subplots()
labels = 'M','F'
ax.pie(rw_dataset['Sex'].value_counts(), labels=labels, autopct='%1.1f%%',startangle=90)
st.pyplot(fig)

#________________________________________________________________________________________________________________
#Upload Clean Dataset......
dfn = pd.read_csv("clnRowDT.csv")
#________________________________________________________________________________________________________________

st.write('Male and Female Who won medal')
st.write("Male 🤴")

fig, ax = plt.subplots()
male = dfn[dfn.Gender == "M"]
m = male.Medal.value_counts()
st.write(m)
ax.pie(male.Medal.value_counts(), labels=m.index, autopct='%1.1f%%',startangle=90)
st.pyplot(fig)

fig, ax = plt.subplots()
st.write("female 👸")
female = dfn[dfn.Gender == "F"]
f = female.Medal.value_counts()
st.write(f)
ax.pie(female.Medal.value_counts(), labels=f.index, autopct='%1.1f%%',startangle=90)
st.pyplot(fig)

st.write("Countries Participated in Olympics from 1896-2016")
t_countries = rw_dataset.region.value_counts()
st.write(t_countries)

st.bar_chart(t_countries)

x = dfn[dfn.Medal == 'Gold'].Region.value_counts().sort_values(ascending=False)
y = dfn[dfn.Medal == 'Silver'].Region.value_counts().sort_values(ascending=False)
z = dfn[dfn.Medal == 'Bronze'].Region.value_counts().sort_values(ascending=False)
# st.write(z)
    
plt.figure(figsize=(20,15))
plt.xticks(rotation=90)

st.write("Countries Won Gold Medals")
st.line_chart(x)


st.write("Countries Won Silver Medals")
st.line_chart(y)


st.write("Countries Won Bronze Medals")
st.line_chart(z)

st.write("""
### Year Wise Disrtibution
""")
yr = dfn[['Year', 'Medal']]
yr.Medal.value_counts()
# st.write(yr)
all = dfn.Year.value_counts()
g = yr[yr.Medal == 'Gold'].Year.value_counts().sort_index(ascending=True)
s = yr[yr.Medal == 'Silver'].Year.value_counts().sort_index(ascending=True)
b = yr[yr.Medal == 'Bronze'].Year.value_counts().sort_index(ascending=True)

st.write("Overall Medals won by particular Year")
st.bar_chart(all)

st.write('Year wise distribution for Gold Medals')
st.bar_chart(g)

st.write('Year wise distribution for Silver Medals')
st.bar_chart(s)

st.write('Year wise distribution for Bronze Medals')
st.bar_chart(b)

#_______________________________________________________________________________________________________________
st.write("""### Season Wise Distribution""")
st.write(" ")
st.write("Participate in Winter and Summer Season")

Sum = rw_dataset[rw_dataset.Season=='Summer']
Win = rw_dataset[rw_dataset.Season=='Winter']

fig = plt.figure(figsize =(20,12))
sns.set(style="darkgrid")

plt.subplot(2,1,1)
sns.countplot(x= 'Year', data=Sum, palette= "Spectral")
plt.title('Participants In Summer Season', fontsize=15)

plt.subplot(2,1,2)
sns.countplot(x= 'Year', data=Win, palette= "Spectral")
plt.title('Participants In Winter Season', fontsize=15)
st.pyplot(fig)


st.write(" ")
st.write("Winner in Winter and Summer Season")

Sum = dfn[dfn.Season=='Summer']
Win = dfn[dfn.Season=='Winter']

fig = plt.figure(figsize =(20,12))
sns.set(style="darkgrid")

plt.subplot(2,1,1)
sns.countplot(x= 'Year', data=Sum, palette= "Spectral")
plt.title('Winner In Summer Season', fontsize=15)

plt.subplot(2,1,2)
sns.countplot(x= 'Year', data=Win, palette= "Spectral")
plt.title('Winner In Winter Season', fontsize=15)
st.pyplot(fig)





womenOlympicsSum = rw_dataset[(rw_dataset.Sex=='F') & (rw_dataset.Season=='Summer')]
womenOlympicsWin = rw_dataset[(rw_dataset.Sex=='F') & (rw_dataset.Season=='Winter')]
a = womenOlympicsSum.Year.value_counts()
b = womenOlympicsWin.Year.value_counts()
st.write('Women Participation In Summer Season')
st.bar_chart(a)

st.write('Women Participation In Winter Season')
st.bar_chart(b)


womenOlympicsSum = dfn[(dfn.Gender=='F') & (dfn.Season=='Summer')]
womenOlympicsWin = dfn[(dfn.Gender=='F') & (dfn.Season=='Winter')]
a = womenOlympicsSum.Year.value_counts()
b = womenOlympicsWin.Year.value_counts()
st.write('Women Medallist In Summer Season')
st.bar_chart(a)

st.write('Women Medallist In Winter Season')
st.bar_chart(b)

st.write("Variation of Male/Female athletes over Time (Summer Games)")
MenOverTimeSum = dfn[(dfn.Gender == 'M') & (dfn.Season == 'Summer')]
WomenOverTimeSum = dfn[(dfn.Gender == 'F') & (dfn.Season == 'Summer')]
MenOverTimeWin = dfn[(dfn.Gender == 'M') & (dfn.Season == 'Winter')]
WomenOverTimeWin = dfn[(dfn.Gender == 'F') & (dfn.Season == 'Winter')]

part1 = MenOverTimeSum.groupby('Year')['Gender'].value_counts()
part2 = WomenOverTimeSum.groupby('Year')['Gender'].value_counts()
fig = plt.figure(figsize=(20, 10))
part1.loc[:,'M'].plot(color='r')
part2.loc[:,'F'].plot(color='b')
plt.title('Variation of Male and Female Medallist over time for Summer Olympics', fontsize=15)
st.pyplot(fig)
part1 = MenOverTimeWin.groupby('Year')['Gender'].value_counts()
part2 = WomenOverTimeWin.groupby('Year')['Gender'].value_counts()
fig = plt.figure(figsize=(20, 10))
part1.loc[:,'M'].plot(color='r')
part2.loc[:,'F'].plot(color='b')
plt.title('Variation of Male and Female Medallist over time for Winter Olympics', fontsize=15)
st.pyplot(fig)


#___________________________________________________________________________________________________
st.write("""### Height vs Weight of Olympic Medalists""")
st.write(" ")
# def scatter_plot():
#     #Create numpy array for the visualisation
#     x = dfn["Height"]
#     y = dfn["Weight"]    
#     fig = plt.figure(figsize=(10, 4))
#     plt.xlabel('Height')
#     plt.ylabel('Weight')
#     plt.scatter(x, y, hue="Medal", style="Medal")
#     st.balloons()
#     st.pyplot(fig)
# scatter_plot()

fig = plt.figure(figsize=(20, 10))
ax = sns.scatterplot(x="Height", y="Weight", data=dfn, hue="Medal", style="Medal")
plt.title('Height vs Weight of Olympic Medalists', fontsize=25)
st.pyplot(fig)

#__________________________________________________________

st.write("Age Distribution Of Medalist Won Medal")
fig, ax = plt.subplots()
ax.hist(dfn["Age"], bins=40)
plt.xlabel("Age")
plt.ylabel("No. of Medallist")
st.pyplot(fig)

#___________________________________________________________________________________________________
st.write("No. of Medals Won in Sports")
sp = dfn.Sport.value_counts().sort_values(ascending=False)
fig = plt.figure(figsize=(10,10))
plt.xlabel("No of Medals")
plt.ylabel("Sports Name")
plt.xticks(rotation=0)
sns.barplot(sp, sp.index)
st.pyplot(fig)


sp_medal = dfn[['Sport', 'Medal']]
sp_medal.Medal.value_counts()
g_medal = sp_medal[sp_medal.Medal == 'Gold'].Sport.value_counts().sort_values(ascending=False).head(50)
s_medal = sp_medal[sp_medal.Medal == 'Silver'].Sport.value_counts().sort_values(ascending=False).head(50)
b_medal = sp_medal[sp_medal.Medal == 'Bronze'].Sport.value_counts().sort_values(ascending=False).head(50)
fig = plt.figure(figsize=(10,20))
#plt.figure(figsize=(20,15))
plt.xticks(rotation=90)

plt.subplot(3,1,1)
plt.title('Won Gold By Top 50 Sports', fontsize=15)
plt.xlabel('Sports', fontsize=15)
plt.ylabel('Gold Medals', fontsize=15)
sns.barplot(x=g_medal, y=g_medal.index, palette = 'rocket')


plt.subplot(3,1,2)
plt.title('Won Silver By Top 50 Sports', fontsize=15)
plt.xlabel('Sports', fontsize=15)
plt.ylabel('Silver Medals', fontsize=15)
sns.barplot(x=s_medal, y=s_medal.index, palette = 'rocket')

plt.subplot(3,1,3)
plt.title('Won Bronz By Top 50 Sports', fontsize=15)
plt.xlabel('Sports', fontsize=15)
plt.ylabel('Bronze Medals', fontsize=15)
sns.barplot(x=b_medal, y=b_medal.index, palette = 'rocket')
plt.tight_layout()
st.pyplot(fig)

st.write("Overall Participant in Olympic from 1896-2016")
participants1 = rw_dataset.Name.value_counts().head(50)
participants2 = dfn.Name.value_counts().head(50)
fig = plt.figure(figsize=(20,20))
plt.subplot(2,1,1)
plt.xticks(rotation=0)
sns.barplot(x = participants1, y = participants1.index, palette="icefire")
plt.title("How many times Participated in Olympics")
# st.pyplot(fig)

plt.subplot(2,1,2)
plt.xticks(rotation=0)
sns.barplot(x = participants2, y = participants2.index, palette="icefire")
plt.title("How many times Won Medals in Olympics")
st.pyplot(fig)








# Upload GDP dataset___________________________________________________________________________________________
dfg = pd.read_csv("dfg.csv")
#___________________________________________________________________________________________

st.write("GDP vs Medals winning Countries in 2016")
X = dfg[(dfg["Year"]==2016) & dfg["Region"]].sort_values(by='GDP',ascending=False)
plt.figure(figsize=(15,15))
plt.xticks(rotation=0)
fig = plt.figure(figsize=(20,20))
a = X.Region
b = X.GDP
c = X.Region.value_counts()

plt.subplot(1,2,1)
plt.title('Country-GDP Distribution',fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel("GDP", fontsize=15)
sns.barplot(X.GDP, X.Region, palette = 'rocket')
plt.subplot(1,2,2)
plt.title('Country won Medals Distribution',fontsize=15)
plt.xlabel('No. of Medals', fontsize=15)
#plt.ylabel("Region", fontsize=15)
sns.barplot(c, c.index, palette = 'rocket')

plt.tight_layout()
st.pyplot(fig)

st.write("GDP vs Medals winning Countries in 2014")
X = dfg[(dfg["Year"]==2014) & dfg["Region"]].sort_values(by='GDP',ascending=False)
plt.figure(figsize=(15,15))
plt.xticks(rotation=0)
fig = plt.figure(figsize=(20,20))
a = X.Region
b = X.GDP
c = X.Region.value_counts()

plt.subplot(1,2,1)
plt.title('Country-GDP Distribution',fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel("GDP", fontsize=15)
sns.barplot(X.GDP, X.Region, palette = 'rocket')
plt.subplot(1,2,2)
plt.title('Country won Medals Distribution',fontsize=15)
plt.xlabel('No. of Medals', fontsize=15)
#plt.ylabel("Region", fontsize=15)
sns.barplot(c, c.index, palette = 'rocket')

plt.tight_layout()
st.pyplot(fig)

st.write("GDP vs Medals winning Countries in 1992")
X = dfg[(dfg["Year"]==1992) & dfg["Region"]].sort_values(by='GDP',ascending=False)
plt.figure(figsize=(15,15))
plt.xticks(rotation=0)
fig = plt.figure(figsize=(20,20))
a = X.Region
b = X.GDP
c = X.Region.value_counts()

plt.subplot(1,2,1)
plt.title('Country-GDP Distribution',fontsize=15)
plt.xlabel('Countries', fontsize=15)
plt.ylabel("GDP", fontsize=15)
sns.barplot(X.GDP, X.Region, palette = 'rocket')
plt.subplot(1,2,2)
plt.title('Country won Medals Distribution',fontsize=15)
plt.xlabel('No. of Medals', fontsize=15)
#plt.ylabel("Region", fontsize=15)
sns.barplot(c, c.index, palette = 'rocket')

plt.tight_layout()
st.pyplot(fig)


st.write("### Thank You....")




# For further Enhancement!!!
# distribute = st.selectbox(
#     "Data Visualization",
#     ("Name", "Gender", "Year", "Region", "Season", "Sport","Medal", "Age & Height")
# )

# classifier = st.selectbox(
#     "Select Classifier",
#     ("Logistic Regression", "SVM", "KNN", "Decision Tree", "Naive Bayes", "XGBoost", "GBC")
# )


# def add_parameter_ui(clf_name):
#     params = dict()
#     if clf_name == "KNN":
#         K = st.sidebar.slider("K",1,15)
#         params["K"] = K
#     elif clf_name == "SVM":
#         C = st.sidebar.slider("C", 0.01, 10.0)
#         params["C"] = C
#     return params

# add_parameter_ui(classifier)
