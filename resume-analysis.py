# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:13:08 2024

@author: Felipe Nogueira
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:28:43 2024

@author: Felipe Nogueira
"""

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


from langchain_core.output_parsers import JsonOutputParser


os.environ["OPENAI_API_KEY"] ='sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

# This is the name of the report which should be in the directory
# You can download the precise PDF I am using from here https://www.pc.gov.pk/uploads/archives/PSDP_2023-24.pdf
name = r'C:\Users\Felipe Nogueira\Downloads\FelipeBarbosaNogueira.pdf'

# This loader uses PyMuPDF
loader_py = PyMuPDFLoader(name)

#This loader uses Unstructured


# Storing the loaded documents as langChain Document object
pages_py = loader_py.load()

text_splitter = CharacterTextSplitter(
    # shows how to seperate
    separator="\n",
    # Shows the document token length
    chunk_size=1000,
    # How much overlap should exist between documents
    chunk_overlap=150,
    # How to measure length
    length_function=len
)

# Applying the splitter
docs = text_splitter.split_documents(pages_py)

# uses OpenAI embeddings to build a retriever
embeddings = OpenAIEmbeddings()
# Creates the document retriever using docs and embeddings
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={'k': 3})

# Define your desired data structure.

class Experience(BaseModel):
    company: str = Field(description="Company worked for")
    period: str = Field(description="Work period")
    duration_months: str = Field(description="Duration of work in months")
    position: str = Field(description="Person's position in the company")

class Education(BaseModel):
    educational_institution: str = Field(description="Educational institution")
    course: str = Field(description="Course of study")
    period: str = Field(description="Course period")
    course_progress: str = Field(description="Course progress status")

class JobAnalysis(BaseModel):
    status: str = Field(description="Candidate's suitability status for the job, fit or unfit")
    justification: str = Field(description="Justification for the suitability classification")

class Candidate(BaseModel):
    name: str = Field(description="Person's name")
    phone: str = Field(description="Person's phone number")
    email: str = Field(description="Person's email")
    address: str = Field(description="Person's address")
    experience: Experience = Field(description="Person's professional experiences")
    education: Education = Field(description="Person's education")
    job_analysis: JobAnalysis = Field(description="Analysis of the candidate's suitability for the job and the justification for this classification")

parser = JsonOutputParser(pydantic_object=Candidate)

# This is the prompt used
template = """

You are a resume analyzer that extracts personal information, 
professional experience (Company, work time, and position), 
history of education of a candidate(institution, course, course time interval, status (completed, ongoing, or incomplete)), analyzes the job requirements, and informs 
if the person is suitable or unsuitable for the position,
as well as the justification for this classification, always returning in JSON format.

query: {query}

{context} 
"""
requiremets='''
For this position, the requirements are for someone who works in the IT sector with experience in data analysis, 
data insights generation, and data science with Python. Knowledge of pandas libraries, Generative AI would be a plus

'''

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(model_name='gpt-4o', temperature=0.7)

chain = (
# The initial dictionary uses the retriever and user supplied query
    {"context":retriever,
     "query":RunnablePassthrough()}
# Feeds that context and query into the prompt then model & lastly 
# uses the ouput parser, do query for the data.
    |  prompt  | model | StrOutputParser() | parser
 
)

#result=chain.invoke('extrair as informacoes de nome,endereco,telefone e email do candidato')


result=chain.invoke('''Extract the candidate's personal information such as name, address, phone number, and email.
                    Additionally, gather information about their professional experiences including the company they worked 
                    for, their position, the period of employment, and the duration in months they worked at each company. 
                    Also, retrieve educational details such as the educational institution, the degree or program pursued 
                    (undergraduate, graduate, or other), the period of study, and the status of the course (completed, ongoing, or incomplete). 
                    Finally, indicate the candidate's classification as either suitable or unsuitable for the position, along with the justification for this classification.
                    {}
                       '''.format(requiremets))