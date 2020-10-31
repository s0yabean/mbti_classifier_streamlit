import streamlit as st

import numpy as np
import pandas as pd
import re
from random import seed

seed(42)
np.random.seed(42)

from sklearn.model_selection import train_test_split
import gensim.downloader as api
from sklearn.neighbors import KNeighborsClassifier
import pickle

def main(): 
    st.title("MBTI Classifier")
    st.header("4 Categories: Analyst, Diplomat, Explorer, Sentinel")
    user_input = st.text_input(label = "Enter text from the person you want to predict MBTI category for:",
                          value = "This book has an interesting framework.")
    #st.write(summarizer_transformer.predict(title))
    
    # input validation
    if len(user_input) < 15:
        st.header("text is too short to make prediction.")
    # api to check that only english is entered

    processed_text = get_only_chars(user_input)
    text_to_embedding = transform_sentence(processed_text, pretrain_embed)
    text_to_embedding = text_to_embedding.reshape(1, -1)
    prediction = knn_model.predict(text_to_embedding.reshape(1, -1))[0]

    st.header("Your predicted MBTI class is:")
    st.header(prediction)
    st.header(labels[prediction])

    st.markdown(description[labels[prediction]])
    model_desc = st.sidebar.selectbox('Model', list(MODEL_DESC.keys()), 0)

@st.cache(allow_output_mutation=True)
def load_models():
    pretrain_embed = api.load('glove-twitter-25')

    infile = open("knnpickle_file", "rb")
    knn_model = pickle.load(infile)
    infile.close()

    return pretrain_embed, knn_model

# Global Variables
pretrain_embed, knn_model = load_models()
labels = {1: "Analyst", 2: "Diplomat", 3: "Explorer", 4: "Sentinel"}
description = {
    "Analyst": """With the Wizards, a good place to start off is to list out your biggest clients in your introduction. \n\nTalk about your achievements and how you are above your competitors. Wizards like to associate with the strong - and this introduction will give them a sense you are someone they should work with. \n\nPrepare an industry analysis and focus on selling the concept of your product, rather than going into details about your product. Wizards tend to like discussing concepts rather than be bogged down by details. Occasionally, they do ask for details, but it is usually so that they can do their own analysis. Be prepared to be challenged, and explain with logical reason why your product is preferred.""",
    "Diplomat""": "The first step to take in connecting with the Healer is to build a personal relationship. Don't focus on trying to sell from the start - rather make effort to connect by being genuinely interested in them. \n\nShow sincerity in your interest.Share your personal belief in the product you are selling and communicate the passion you have for the product. Even though your passion may be different from theirs, they enjoy seeing someone who is deeply certain about what they stand for. If it is relevant, talk about your product elevants the mental, emotional and spiritual well-being of others - Healers, as their name suggests, want to see the world become better and humanity lifted up.""",
    "Explorer""": "With the Explorers, you don't have to be too serious. Be friendly, easy-going and casual and make some small talk with them. They're most open to you when things are light and easy. \n\nDuring the pitch, keep it short and keep them engaged by having a dialogue. Explorers tend to disengage quickly if they are asked to just sit down and listen to a presentation.  \n\nIf possible, make the presentation appealing to all the five senses - when they experience a product or service and they like it, nothing much else has to be said. \n\nThey are already likely to buy.""",
    "Sentinel""": "You should take a straightforward, direct approach when engaging the Knights. \n\nSet the agenda from the start and share about your years of experience in the industry and any certifications you have from authoritative bodies.Be prepared and organized with the facts - Knights often request for details of the product and they would like to compare your product to other similar products as well so that they know they're getting the best value for money. \n\nTo support the value of your product, remember to back up your pitch with statistics, hard data and real life testimonials."""
    }
MODEL_DESC = {
    "KNN (K-Nearest Neighbors) Classifier" : "Pretrained glove-twitter-25 trained on 1200 twitter users with self-identified MBTI categories."
}

def transform_sentence(text, model):
    """
    Mean embedding vector
    """
    def preprocess_text(raw_text, model=model):

        """ 
        Excluding unknown words and get corresponding token
        """
        raw_text = raw_text.split()
        return list(filter(lambda x: x in model.vocab, raw_text))

    tokens = preprocess_text(text)
    if not tokens:
        return np.zeros(model.vector_size)
    text_vector = np.mean(model[tokens], axis=0)
    return np.array(text_vector)

def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

# dataframe = np.random.randn(10, 20)
# st.dataframe(dataframe)

# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
# st.table(dataframe)

# x = st.slider('x')  # 👈 this is a widget
# st.write(x, 'squared is', x * x)

# # Add a selectbox to the sidebar:
# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# # Add a slider to the sidebar:
# add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )

# left_column, right_column = st.beta_columns(2)
# # You can use a column just like st.sidebar:
# left_column.button('Press me!')

# # Or even better, call Streamlit functions inside a "with" block:
# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
#     st.write(f"You are in {chosen} house!")

if __name__ == '__main__':
    main()
