# SQL HR Query App

This is a web application built with Streamlit that allows users to query a SQL database using natural language questions.

## Features

- Select a table from the dropdown menu.
- Enter a question about the data using natural language.
- Execute the query based on the question.
- View the query results as a table.
- Download the query results as a CSV file.
- Repeat the process to ask additional questions.

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/sql-query-app.git
   ```

2. Change into the project directory:

   ```shell
   cd sql-query-app
   ```

3. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

## Usage

1. Make sure you have a SQLite database file (*.db) or provide the appropriate database connection URI.
2. Run the Streamlit app:

   ```shell
   streamlit run app.py
   ```

3. Access the app in your web browser at `http://localhost:8501`.

## Technologies Used

- Python
- Streamlit
- SQLAlchemy
- LangChain

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This app was developed as part of a project and is based on the [LangChain](https://github.com/langchain/langchain) library.
