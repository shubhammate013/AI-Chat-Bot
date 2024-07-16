
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All


@st.cache_resource
def load_model():
    model_path = "./models/mistral-7b-openorca.Q4_0.gguf"
    callback_manager = CallbackManager([])
    llm = GPT4All(model=model_path,
                  callback_manager=callback_manager, verbose=True)
    return llm


@st.cache_resource
def load_vectorstore():
    loader = DirectoryLoader('data/', glob="**/*.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = GPT4AllEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore


def main():
    st.title("GPT4All Chatbot")
    llm = load_model()
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    query = st.text_input("Enter your question:")
    if st.button("Ask"):
        result = qa.invoke(input=query)
        st.write(result["result"])


if __name__ == "__main__":
    main()
