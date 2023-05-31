# Import necessary libraries
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
import streamlit as st
from sqlalchemy import create_engine, inspect
import pandas as pd 
import base64


# Connect to the database
db_uri = "sqlite:////Users/shahryar/Desktop/SQLite3/hr_demo.db"
engine = create_engine(db_uri)
inspector = inspect(engine)
table_names = inspector.get_table_names()

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
    col1, col2 = st.columns(2)
    with col2:
        st.write("## Select a table to query")
        selected_table = st.selectbox("Choose a table", options=table_names)
        if selected_table:
            query = f"SELECT * FROM {selected_table} LIMIT 1"
            df = pd.read_sql_query(query, engine)
            # Display first row
            st.write("Below, you can see what columns are available to use:")
            st.dataframe(df)
    with col1:


        # Streamlit interface
        st.header('Instructions for using the SQL Query App')

        st.header('Table Selection')
        st.write('1. Choose a table from the dropdown menu. This will determine the table to be queried.')

        st.header('Ask a Question')
        st.write('2. Type your question about the data in the text input field. The question should be in a natural language format, such as "How many employees are there?" or "What is the average salary?".')

        st.header('Run Query')
        st.write('3. Click the "Run Query" button to execute the query based on your question.')

        st.header('View Results')
        st.write('4. If the query generates table-like results, the app will display the results as a table. You can scroll through the table to view all the rows and columns.')

        st.header('Download Results')
        st.write('5. If the query generates table-like results, a "Download CSV File" button will appear below the table. Clicking this button will download the query results as a CSV file named "query_results.csv".')

        st.header('Ask Another Question')
        st.write('6. You can repeat the process by selecting a different table, entering a new question, and clicking the "Run Query" button again.')


    with col2:
        st.write("## Ask a question about the data")
        user_question = st.text_input("Type your question here...")
        
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
