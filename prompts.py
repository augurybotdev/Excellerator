from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate,HumanMessagePromptTemplate
import datetime
import time
from langchain.chat_models import ChatOpenAI
import os

from apikey import apikey

os.environ['OPENAI_API_KEY'] = apikey



model = "gpt-3.5-turbo"
chat_model =ChatOpenAI(model=model)


def answer_query(data, question):
    
    template="""Examine the data and answer the following question based on the data:
    data:
    {data}
    question:
    {question}
    your response:        
    """
    prompt = PromptTemplate(
        input_variables=["data", "question"],
        template=template
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=prompt
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chat_model = ChatOpenAI(model=model, temperature=0.9)
    chain= LLMChain(llm=chat_model, prompt=chat_prompt_template, verbose=True)
    input = {"data": data, "question": question}
    response = chain.predict(data=data, question=question)
    answer = str(response)
    return answer


def evaluate_query_data(statistics, question):
    
    template="""The user's data set has been deemed too large to process.
    please analyze the user's query along with some statistics from the over-sized data-set.
    consider whether there is a means by which you can extract the meaningful data needed in order to answer the user's query.
    Here are some statistics about the data:
    {statistics}
    and here is the user's query:
    {query}
    if the user's query is answerable from the statistics, please do so. If not, respond with this exact message and this exact message only:
    st.write("proceed with chunking of data-set")
    your response:
    
    """
    prompt = PromptTemplate(
        input_variables=["data", "question"],
        template=template
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=prompt
    )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chat_model = ChatOpenAI(model=model, temperature=0.9)
    chain= LLMChain(llm=chat_model, prompt=chat_prompt_template, verbose=True)
    input = {"statistics" :  statistics, "question": question}
    response = chain.predict(statistics=statistics, question=question)
    answer = str(response)
    return answer

