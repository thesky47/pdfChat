import streamlit as st
import os 
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat PDF App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made by Akash Bindu')

def main():
    load_dotenv()
    os.environ["FAISS_NO_AVX2] = False
    st.header("Chat PDF App")

    if pdf := st.file_uploader("Upload your PDF", type="pdf"):
        pdf_reader = PdfReader(pdf)

        text = "".join(page.extract_text() for page in pdf_reader.pages)

        # LLM have limited context window 
        # splitting text due to restrictions on context tokens in openAI (chatGPT allow only 4096 tokens)


        splitter =  RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text) #split_documents

        # Embedding text
        # creating vectors from works to create vector store useful for searching 
        # getting similarity between words
        # creating vector store using openAI 
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)


        else:
            encode_kwargs = {'normalize_embeddings': True}

            # local model to save cost of openAi
            # model = HuggingFaceInstructEmbeddings(model_name='BAAI/bge-base-en',
            #                                     # retrieval passages for short query, using query_instruction, else set it ""
            #                                     query_instruction="Represent this sentence for searching relevant passages: ",
            #                                     model_kwargs = encode_kwargs
            #                                     )

            model = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=model) # from documents
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)


        if query := st.text_input("Ask questions about your PDF file:"):
            docs = vector_store.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
                st.write(response)
        

        #st.write(chunks)



if __name__ == '__main__':
    main()
