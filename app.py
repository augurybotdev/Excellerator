import os
import streamlit as st
import pandas as pd
import base64
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from prompts import answer_query, evaluate_query_data
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter
from langchain.callbacks import get_openai_callback
from apikey import apikey
from langchain.document_loaders.csv_loader import CSVLoader

os.environ['OPENAI_API_KEY'] = apikey
model = "gpt-3.5-turbo"
chat_model =ChatOpenAI(model=model)
llm = OpenAI(model_name="gpt-3.5-turbo")


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()# some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="edited_dataframe.csv">Download CSV File</a>'
    return href

def query_submit_callback():
    st.session_state['is_query_submitted'] = True

st.markdown("# EXCEL-ERATOR")


tab1, tab2 = st.tabs(["Edit Uploaded CSV File", "AI Insights"])
uploaded_file = tab1.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Now we can give the path to CSVLoader
    loader = CSVLoader("temp.csv")
    lang_data = loader.load()

    # Convert the list of dictionaries to a string
    lang_data_str = "\n".join([str(row) for row in lang_data])
    
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Now we can give the path to CSVLoader
    loader = CSVLoader("temp.csv")
    lang_data = loader.load()
    # Convert the list of dictionaries to a string
    lang_data_str = "\n".join([str(row) for row in lang_data])
    # Estimate the number of tokens in the string
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=4096, chunk_overlap=0)
    split_texts = text_splitter.split_text(lang_data_str)
    estimated_token_count = sum([len(text) for text in split_texts])
    st.write(f"The CSV file is estimated to contain {estimated_token_count} tokens.")
    # Split the string into chunks
    split_texts = text_splitter.split_text(lang_data_str)
    print(f"The CSV file was split into {len(split_texts)} chunks.")
    print(f"The first chunk is: {split_texts[0]}")
    if not isinstance(data, pd.DataFrame):
        st.error("The uploaded file could not be read as a DataFrame.")
    elif data.empty:
        st.error("The uploaded file is empty.")
    else:
        # Allow the user to edit the DataFrame
        tab1.write('Edit DataFrame')
        edited_data = tab1.data_editor(data, num_rows="dynamic")
        csv_string = edited_data.to_csv(index=False)
        # Initialize the history in the session state
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
#... other code
#... other code
        if "is_query_submitted" not in st.session_state:
            st.session_state['is_query_submitted'] = False
        form = tab2.form(key="query_submit")
        query = form.text_input("Enter your natural language query:", value="")
        if query:
            st.session_state['query'] = query  # Store the query in st.session_state
            split_query_texts = text_splitter.split_text(query)
            estimated_query_token_count = sum([len(text) for text in split_query_texts])
            form.write(f"Query = {estimated_query_token_count} tokens.")
            form.write(f"Data-set = {estimated_token_count} tokens.")
            total_estimated_token_count = estimated_query_token_count+estimated_token_count
            form.write(f"Total tokens = {total_estimated_token_count} tokens.")
        if form.form_submit_button("Submit Query", on_click=query_submit_callback):
            st.session_state['is_query_submitted'] = True
        if st.session_state['is_query_submitted']:
            query = st.session_state['query']  # Retrieve the query from st.session_state
            #... your try block of code
#... the rest of my code

            try:
                answer = answer_query(csv_string, query)
                tab2.write(answer)
            except Exception as e:
                try:
                    tab2.write(str(e))  # Print out the error message
                        
                    # ... rest of your code
                    column_names = edited_data.columns.tolist() 
                    # Extract row labels
                    row_labels = edited_data.index.tolist() 
                    # Calculate basic statistics
                    statistics = edited_data.describe() 
                    # Check for missing values
                    missing_values = edited_data.isnull().sum() 
                    # Identify unique values for categorical columns
                    unique_values = {column: edited_data[column].unique() for column in edited_data.columns if edited_data[column].dtype == 'object'}
                    evaluation = evaluate_query_data(query, statistics)
                    # Display the extracted information
                    tab2.write(column_names)
                    tab2.write(row_labels)
                    tab2.write(statistics)
                    tab2.write(missing_values)
                    tab2.write(unique_values) 
                    # Provide suggestions based on the extracted information
                    line1 = tab2.write("The dataset is too large for the AI to analyze in its entirety. Here are some suggestions:")
                    line2 = tab2.write("- Try focusing on a subset of the data. For example, you could select a specific column or range of rows to analyze.")
                    line3 = tab2.write("- Try aggregating the data. For example, if the data is time-series data, you could aggregate it by week, month, or year.")
                    line4 = tab2.write("- Try cleaning the data. For example, you could fill in or remove missing values.")
                    line5 = tab2.write("- Try simplifying the data. For example, if a categorical column has a lot of unique values, you could group similar values together.")
                    answer = [line1, line2, line3, line4, line5]
                except Exception as e:
                    new_answer = st.write(f"ERROR: {e}")
            # Append the current query and answer to the history
            st.session_state['history'].append((query, answer))
            next_action = tab2.button("save results")
            if next_action:
                history_expander = tab1.expander("History")
                with history_expander:
                    # Loop over the history and display each query and answer
                    for i, (previous_query, previous_answer) in enumerate(st.session_state['history'], start=1):
                        st.write(f"Query {i}: {previous_query}")
                        st.write(f"Answer {i}: {previous_answer}")