# Import necessary libraries
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import streamlit as st
from sqlalchemy import create_engine, inspect
import pandas as pd

# Connect to the database
db_uri = "sqlite:////Users/shahryar/Desktop/hr_demo.db"
engine = create_engine(db_uri)
inspector = inspect(engine)
table_names = inspector.get_table_names()

# Initialize the language model and the database chain
llm = OpenAI(temperature=0, verbose=True)
db = SQLDatabase.from_uri(db_uri)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# App title and introduction
st.title('HR Database Q&A Demo')
st.markdown('This app translates natural language questions into SQL queries and executes them on a database. Simply type in a question and see the SQL query and its result.')

st.write("## Select a table to query")
selected_table = st.selectbox("Choose a table", options=table_names)

st.write("## Ask a question about the data")
user_question = st.text_input("Type your question here...")

# Translation and execution button

if st.button("Submit"):
    if user_question:
        result = db_chain.run(user_question)
        # df = pd.DataFrame(result, columns=["Your", "Columns", "Here"])  # Replace "Your", "Columns", "Here" with your actual column names
        st.write("## Result")
        st.write(result)
        # st.dataframe(df)


