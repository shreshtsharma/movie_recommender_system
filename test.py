# %%
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_icon="🎬",page_title="Movie recommender")

# %%
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

# %%
# movies.head()

# %%
# credits.head()

# %%
#merging both dataframes
movies=movies.merge(credits,on='title')

# %%
# movies.info()

# %%
#extracting selected columns which i want to use 
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# %%
# movies.head()

# %%
# movies.isnull().sum()

# %%
#droping movies whch have null values in overview
movies.dropna(inplace=True)

# %%
#checking duplicate values in dataset
# movies.duplicated().sum()

# %%
# movies.iloc[0].genres

# %% [markdown]
# function tolist to extract only useful index of dictionaries like only name from genres dictionary

# %%

import ast
#importing ast to convert string which we are getting as input
# into a list so that we can iterate over it.
def tolist(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# %%

movies['genres']=movies['genres'].apply(tolist)
# movies.head()

# %%
movies['keywords']=movies['keywords'].apply(tolist)
# movies.head()

# %%
# movies['cast'][0]

# %%

#function to find top four cast members in a movie
import ast
def maincast(obj):
    L=[]
    count=0
    for i in ast.literal_eval(obj):
        if(count<4):
            L.append(i['name'])
            count+=1
        else:
            break
    return L


# %%
movies['cast']=movies['cast'].apply(maincast)
# movies['cast'][0]

# %%
# movies.head()

# %%
#function to extract only director name from crew column
def director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            L.append(i['name'])
            break
    return L

# %%
movies['crew']=movies['crew'].apply(director)
# movies.head()

# %%
#to convert overview from string to a list so that we can concatenate it with other columns to get tags for a movie
movies['overview']=movies['overview'].apply(lambda i:i.split())
# movies.head()

# %%
#removing spaces from each column because it can create ambiguity
movies['genres']=movies['genres'].apply(lambda x:[ i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
# movies.head()

# %%
#making another column which is tags which is basis of our reccomendation
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
# movies.head()


# %%
movies_new=movies[['movie_id','title','overview','tags']]
# movies_new.head()

# %%
#converting list to string in tags column
movies_new['tags']=movies_new['tags'].apply(lambda x:" ".join(x))


# %%
# movies_new.head()

# %%
movies_new['overview']=movies_new['overview'].apply(lambda x:" ".join(x))

# %%
# movies_new.head()

# %%
# movies_new['tags'][0]

# %%
#converting whole string to lowercase
movies_new['tags']=movies_new['tags'].apply(lambda x:x.lower())

# %%
# movies_new.tail()

# %% [markdown]
# preprocessing of data is done

# %% [markdown]
# BUILDING MAIN MODEL 

# %%
#performing stemming on our tags so that words with similar meaning can't be considered as different words
import nltk
from nltk.stem.porter import PorterStemmer
#making an object of porter scanner.
ste=PorterStemmer()
def stemtag(tag):
    L=[]
    for i in tag.split():
        L.append(ste.stem(i))

    new_tag=" ".join(L)
    return new_tag



# %%
movies_new['tags']=movies_new['tags'].apply(stemtag)
# movies_new['tags'][0]
# movies_new.tail()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
#creating a object cv which will help in vectorization
cv=TfidfVectorizer(max_features=5000,stop_words='english')

# %%
vectors=cv.fit_transform(movies_new['tags']).toarray()
#.toarray is used to convert sparse matrix which is returned 
#by countvectorizer function into a numpy array
# 
# #(vectors)

# %%
# cv.get_feature_names()

# %%
# cosine_similarity will calculate similarity of each movie with every movie
from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
# similarity is a variable which contains similarity of each movie with others
# similarity

# %% [markdown]
# MAIN FUNCTION WHICH WILL FETCH US SIMILAR MOVIES FOR A GIVEN MOVIE

# %%
# enumerate functions helps us to have index position of every movie
list(enumerate(similarity[0]))


# %%
# movies_new['title']=movies_new['title'].apply(lambda x:x.lower())

# %%
def recommend(movie):
    movie_index=movies_new[movies_new['title']==movie].index[0]
    cos_distance=similarity[movie_index] 
    # after this we will sort the similarity with their index and then we will reccommend top 10 similar movies
    movies_list=sorted(list(enumerate(cos_distance)),reverse=True,key=lambda x:x[1])
    movies_list=movies_list[1:11]
    
    # for i in movies_list:
        # print(i[0])
    #    print(movies_new.iloc[i[0]].title)

# %%
# movies_new.shape
movies_new=movies_new.truncate(after=4804)
# movies_new.tail()

# %%

recommend('Toy Story')

# %%
# import pickle
# pickle.dump(movies_new,open('movies.pkl','wb'))
# pickle.dump(similarity,open('similarity.pkl','wb'))

# %%
# movies_new.head()


import pickle
import requests



# st.set_page_config(layout="wide", page_icon="🎬",page_title="Movie recommender")
# movies=pickle.load(open('movies.pkl','rb'))
# similarity=pickle.load(open('similarity.pkl','rb'))

movies_title=movies_new['title'].values

st.title('Movie Recommender System')
st.subheader('Recommends movie based on content')

def poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=708a09b5a0ab52661997cc2b1070ecad&language=en-US'.format(movie_id))
    data=response.json()
    if data['poster_path']==None:
        return data['poster_path']
    else:
         return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def recommend(movie):
    movie_index=movies_new[movies_new['title']==movie].index[0]
    cos_distance=similarity[movie_index] 
    # after this we will sort the similarity with their index and then we will reccommend top 20 similar movies
    movies_list=sorted(list(enumerate(cos_distance)),reverse=True,key=lambda x:x[1])
    movies_list=movies_list[1:11]

    recommended=[]
    recommended_posters=[]
    movie_overview=[]
    for i in movies_list:
        movie_id=movies_new.iloc[i[0]].movie_id #movies id which are recommended is stored in movie_id to fetch their posters
        
        recommended.append(movies_new.iloc[i[0]].title)
          #fetching posters from API
        movie_overview.append(movies_new.iloc[i[0]].overview)
        recommended_posters.append(poster(movie_id))
        # recommended_posters
        # poster(movie_id)
    return recommended,recommended_posters,movie_overview

selected_movie=st.selectbox("enter movie name",movies_title)

if st.button ("Recommend movies"):
    
    recommendations,posters,overview=recommend(selected_movie)
     
    st.subheader("These are the top 10 similar movies related to your search:")
    
    col1,col2,col3,col4=st.columns(4)
    with col1:
        if posters[0]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[0])
        st.write(recommendations[0])
        with st.expander("Overview",expanded=False):
            st.info(overview[0])
    with col2:
        if posters[1]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[1])
        st.write(recommendations[1])
        with st.expander("Overview",expanded=False):
            st.success(overview[1])
    with col3:
        if posters[2]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[2])
        st.write(recommendations[2])
        with st.expander("Overview",expanded=False):
            st.warning(overview[2])
    with col4:
        if posters[3]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[3])
        st.write(recommendations[3])
        with st.expander("Overview",expanded=False):
            st.error(overview[3])
    col5,col6,col7,col8=st.columns(4)
    with col5:
        if posters[4]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[4])
        st.write(recommendations[4])
        with st.expander("Overview",expanded=False):
            st.success(overview[4])
    with col6:
        if posters[5]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[5])
        st.write(recommendations[5])
        with st.expander("Overview",expanded=False):
            st.error(overview[5])
    with col7:
        if posters[6]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[6])
        st.write(recommendations[6])
        with st.expander("Overview",expanded=False):
            st.info(overview[6])
    with col8:
        if posters[7]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[7])
        st.write(recommendations[7])
        with st.expander("Overview",expanded=False):
            st.warning(overview[7])
    col9,col10,col11,col12=st.columns(4)
    with col9:
        
        if posters[8]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[8])
        st.write(recommendations[8])
        with st.expander("Overview",expanded=False):
            st.warning(overview[8])
    with col10:
        if posters[9]==None:
            st.image("images.png",width=255)
        else:
            st.image(posters[9])
        st.write(recommendations[9])
        with st.expander("Overview",expanded=False):
            st.success(overview[9])
   
   
st.write("Made by : Shresht sharma")