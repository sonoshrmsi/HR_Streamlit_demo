# Import necessary libraries
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain import PromptTemplate, LLMChain
import streamlit as st
from sqlalchemy import create_engine, inspect
import pandas as pd 
import base64
import os

from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain. vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI



# Connect to the database
db_uri = f"sqlite:///{os.getcwd()}/hr_demo.db"
engine = create_engine(db_uri)
inspector = inspect(engine)
table_names = inspector.get_table_names()

# Langchain Chain
chain = load_qa_chain(OpenAI(), chain_type='stuff')


# Initialize the language model and the database chain
llm = OpenAI(temperature=0, verbose=True)
db = SQLDatabase.from_uri(db_uri)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)


# STREAMLIT APP LAYOUT
st.markdown("""
    <style>
        .reportview-container {
            padding: 10 !important;
        }
        .main .block-container {
            max-width: 90%;
        }
    </style>
    """, unsafe_allow_html=True)

# App title and introduction

with st.container():
    st.title('HR Database Q&A Demo')
    st.markdown('This app translates natural language questions into SQL queries and executes them on a database. Simply type in a question and see the SQL query and its result.')

with st.container():
    st.write("## Select a table to query")
    selected_table = st.selectbox("Choose a table", options=table_names)
    if selected_table:
        query = f"SELECT * FROM {selected_table} LIMIT 1"
        df = pd.read_sql_query(query, engine)
        # Display first row
        st.write("Below, you can see what columns are available to use:")
        st.dataframe(df)
    st.write("## Ask a question about the data")
    _user_question = st.text_input("Type your question here...")
    user_question = f"{_user_question}. Make sure to return all the columns"
    # Translation and execution button
    if st.button("Submit"):
        if user_question:
            result = db_chain(user_question)
            st.write("## Result")      
            st.write(result["intermediate_steps"][5])
            
            # showing the result if there is a table
            result_list = eval(result["intermediate_steps"][3])
            if len(result_list[0]) > 1:

                column_names = inspector.get_columns(selected_table)
                column_names = [col["name"] for col in column_names]
                df = pd.DataFrame(result_list, columns=column_names)
                # st.write("## Result")
                # st.dataframe(df)
                # Download button
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="query_results.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.write("## Result")
                st.dataframe(df)


                embeddings = OpenAIEmbeddings()
                docsearch = FAISS.load_local('docsearch_index/', embeddings)

                # LLM creativity
                llm = OpenAI(temperature=0.5)
                prompt_template = "Based on the following piece of text: {search_result}\n\n provide some background information to the following question: \n {question}\n\n If you feel the text is not relevant to the question, don't mention it, ignore the text, and give some background information as you see fit."
                prompt = PromptTemplate(template=prompt_template, input_variables=["search_result", "question"])

                llm_chain = LLMChain(prompt=prompt, llm=llm)
        
                search_result = docsearch.similarity_search(user_question)

                print(search_result)
                answer = llm_chain.run({'search_result': search_result,
                                        "question": user_question})
                
                st.write(answer)
