from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader


model_path = "./models/mistral-7b-openorca.Q4_0.gguf"
callback_manager = CallbackManager([])
llm = GPT4All(model=model_path,
              callback_manager=callback_manager, verbose=True)

template = """
You are an AI assistant.
Given the following question, provide a detailed answer.

Question: {question}

Answer:
"""


prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
# query = "What are the benefits of using GPT4All?"
# result = llm_chain.invoke(input=query)


# print(result["text"])

# take from user input
# while True:
#     query = input("Enter your question: ")
#     result = llm_chain.invoke(input=query)
#     print(result["text"])


loader = DirectoryLoader('data/', glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# Create a vector store and add the text chunks
embeddings = GPT4AllEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Load the question-answering chain with the retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)


# Ask a question and get the answer
query = "What is in your knowledge base?"
result = qa.invoke(input=query)
print(result["result"])
