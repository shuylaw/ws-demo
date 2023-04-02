import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
import json
import pandas as pd

template = """
This is a long keyword "Reinforcement learning"
This is a short keyword "Reinforcement"

This is a long keyword "Logistic Regression"
This is a short keyword "Regression"

This is a long keyword "K-Nearest-Neighbor"
This is a short keyword "KNN"

Given a topic, generate relevant keywords that are short and fewer than 13 characters each, grouped by subtopics. \
There should be at least 10 subtopics. Each subtopic should have 10 or more keywords consisting of one or two words that are complete words. \
Proper nouns are allowed if they are relevant to the subtopic.

Example Topic: Artificial Intelligence
Example of good keywords for the subtopic Machine Learning:
1) Algorithm
2) Dataset
3) Pandas
4) Numpy
5) KNN
6) Naive Bayes

Example of bad keywords for the subtopic Machine Learning:
1) Reinforcement learning (longer than 13 characters)
2) Feature sel (abbreviation)
3) Logistic Regression (longer than 13 characters)
4) Knowledge representation and reasoning (more than two words)
5) Voice activity detection (more than two words)
6) Support vec (abbreviation)

Your response should be in the JSON format:
{{"Subtopic": ["Keyword1", "Keyword2", "Keyword3", "Keyword4", "Keyword5", "Keyword6", "Keyword7", "Keyword8", "Keyword9", "Keyword10"]}}

Your response below for the topic {topic}:
"""

prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)

def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=1, openai_api_key=openai_api_key, max_tokens=1000)
    return llm


st.set_page_config(page_title="Word Search Word List Generator", page_icon=":pencil2:")
st.header("Word Search Word List Generator")

col1, col2 = st.columns(2)

with col1:
    st.markdown("A word search puzzle is a game where words are hidden in a grid of letters, and the objective is to find and circle them. \
                \n\nWord search puzzle books usually contain thematic puzzles, each with lists of words relevant to a topic. \
                Although lists of words can be easily obtained through existing methods (web scraping, buying pre-made word lists, etc), the words are usually not \
                grouped into subtopics. Puzzle books without subtopic puzzles (see image) create a word search experience that is less than ideal for the end user. \
                \n\nCurating good lists of words that are properly categorised into subtopics requires quite a bit of manual work. \
                This mini app attemps to automate the creation of these subtopic word lists by generating them with text-davinci-003.")

with col2:
    st.image(image="xmas-ws-ans.png", width=300, caption="A Christmas puzzle with words that are vaguely related to Christmas (no subtopic grouping)")

st.markdown("### Enter a topic")

def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

openai_api_key = get_api_key()

def get_topic():
    input_text = st.text_input(label="Enter a topic",  placeholder="Christmas", key="ws_topic")
    return input_text

def create_df(data):
    """ Turns JSON data into a dataframe """
    dict = json.loads(data)
    df = pd.DataFrame(dict)
    df.index += 1
    return df

topic_input = get_topic()

if topic_input:
    if not openai_api_key:
        st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
        st.stop()

    llm = load_LLM(openai_api_key=openai_api_key)

    prompt_with_topic = prompt.format(topic=topic_input)

    response = llm(prompt_with_topic)

    df = create_df(response)
    st.write(df)
    st.write(response)

st.markdown("### Example Lists")
st.markdown("Use the checkboxes below to show pre-generated example lists")

xmas_data = '''
{ "Decorating": ["Tree", "Lights", "Ornaments", "Garlands", "Mistletoe", "Stockings", "Wreath", "Sleigh", "Fireplace", "Candles"], "Gifts": ["Dress", "Hat", "Socks", "Pajamas", "Bracelet", "Book", "Toy", "Game", "Candy", "Wallet"], "Food": ["Gingerbread", "Cake", "Pie", "Punch", "Cookies", "Candy", "Pudding", "Roast", "Fruitcake", "Spices"], "Music": ["Jingle Bells", "Joy to the World", "O Christmas Tree", "Silent Night", "Deck the Halls", "Let It Snow", "Carol of the Bells", "We Wish You a Merry Christmas", "Rudolph the Red-Nosed Reindeer", "Hark! The Herald Angels Sing"], "Events": ["Tree Lighting", "Caroling", "Parade", "Pageant", "Secret Santa", "Gift Exchange", "Family Gathering", "Cabin Trips", "Sleigh Ride", "Concert"], "Crafts": ["Gift Bags", "Snow Globes", "Paper Angels", "Yarn Strings", "Wreath Bows", "Stars Ornaments", "Candy Canes", "Popsicle Trees", "Cookie Cutters", "Card Making"], "Traditions": ["Present Opening", "Gingerbread House", "Christmas Carols", "Tinsel Usage", "Merry Toast", "Kiss Under Mistletoe", "Letter to Santa", "White Christmas", "Stocking Stuffing", "Yule Log"], "Symbols": ["Santa Clause", "Reindeers", "Christmas Tree", "Wise Men", "North Pole", "Snowman", "Elves", "Snow Angels", "Angels", "Candy Canes"], "Holidays": ["Thanksgiving", "Hanukkah", "Kwanzaa", "Winter Solstice", "New Year's", "Christmas Eve", "Christmas", "Boxing Day", "St Steven's Day", "Epiphany"], "Activities": ["Caroling", "Sledding", "Skiing", "Ice Skating", "Snowball Fights", "Shopping", "Serving", "Gingerbread House", "Cocoa Drinking", "Movie Night"] }
'''

dog_data = '''
{"Domestic Breeds": ["Labrador", "Poodle", "Pug", "Golden Retriever", "Beagle", "Bulldog", "Border Collie", "Shih Tzu", "Pomeranian", "Husky"], "Therapy Dogs": ["Service Dog", "Comfort Dog", "Psychiatric Service Dog", "Service Animal", "Therapy Pet", "Facility Dog", "Animal Assisted Therapy", "Canine Therapy", "Animal Companion", "Human-Animal Bond"], "Rescue Dogs": ["Adoption", "Socialization", "Foster Care", "Shelter Dog", "Rescue Group", "Spay/Neuter", "Adoption Fees", "Animal Control", "Volunteer Work", "Feral Dog"], "Working Dogs": ["Guard Dog", "Herding Dog", "Sledding Dog", "Search & Rescue", "Cadaver Dog", "Police Dog", "Bomb Detection", "Disability Assistance", "Therapy Dog", "Hunting Dog"], "Hybrid Breeds": ["Maltipoo", "Labradoodle", "Golden Doodle", "Puggle", "Cockapoo", "Goldendoodle", "Bullpug", "Pomapoo", "Corgi Poodle", "Cockerpoo"], "Toy Breeds": ["Yorkshire Terrier", "Papillon", "Chihuahua", "Affenpinscher", "Toy Poodle", "Pekingese", "Maltese", "Shih Tzu", "Pomeranian", "Brussels Griffon"], "Rare Breeds": ["Lagotto Romagnolo", "Thai Ridgeback", "Tamaskan Dog", "Komondor", "Havana Silk Dog", "Dongo Venezuelano", "Terceira Cattle Dog", "Central Asian Shepherd", "Caucasian Shepherd Dog", "Chinook"], "Sporting Dogs": ["Retrievers", "Spaniels", "Setters", "Pointers", "Labradors", "Vizslas", "Weimaraners", "German Shorthaired Pointers", "Flat-coated Retrievers", "Chesapeake Bay Retrievers"] }
'''

psych_data = '''
{"Cognitive Processes": ["Memory", "Thinking", "Reasoning", "Decision-Making", "Creativity", "Beliefs", "Perception", "Attention", "Problem-Solving", "Judgement"], "Emotion and Motivation": ["Happiness", "Anxiety", "Frustration", " Social motivation", "Boredom", "Aggression", "Self-Esteem", "Love", "Moods", "Stress"], "Cognitive Development": ["Piaget", "Stages", "Metacognition", "Vygotsky", "Imitation", "Adolescence", "Brain", "Schemas", "Growth", "Cognitive Wrk"], "Personality Development": ["Adler", "Behaviour", "Traits", "Freud", "Inferiority", "Egoism", "Consciousness", "Culture", "Morality", "Self-Actualization"], "Abnormal Psychology": ["Insomnia", "Anorexia", "Psychosis", "Pathology", "Hysteria", "Somatoform", "Clinical Profile", "Depression", "Psychopharmacology", "Mania"], "Cognitive Disorders": ["Dementia", "Schizophrenia", "Amnesia", "Alzheimer's", "Aphasia", "Anosognosia", "Autism", "Disorganized", "Catatonia", "Psychosis"], "Social Psychology": ["Attitudes", "Roles", "Groups", "Socialization", "Relationships", "Influence", "Prejudice", "Discrimination", "Altruism", "Social Norms"]}
'''

col1, col2 = st.columns([1, 3], gap="medium")

with col2:
    xmas_placeholder = st.empty()
    dog_placeholder = st.empty()
    psych_placeholder = st.empty()

with col1:
    xmas_df = create_df(xmas_data)
    dog_df = create_df(dog_data)
    psych_df = create_df(psych_data)

    if st.checkbox("Show Christmas List"):
        xmas_placeholder.dataframe(xmas_df)

    if st.checkbox("Show Dog List"):
        dog_placeholder.dataframe(dog_df)

    if st.checkbox("Show Psychology List"):
        psych_placeholder.dataframe(psych_df)