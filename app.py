from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask your PDF 💬")
    
    # upload pdf using streamlit
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Read/parse the uploaded pdf to extract text
    if pdf is not None:
        pdfReader = PdfReader(pdf)
        # Loop thorugh pages of pdf
        text = ""
        for page in pdfReader.pages:
            text+=page.extract_text()
        
        # Problem - The extract text is huge an cannot be fed to langchain to answer user's questions.
        # Solution - split  it to similar size chunks
        
        # Split into chunks
        
        
        # chunk_size = size of chunk
        # chunk_overlap = no of characters that can overlap between two chunks
        # length_function = function to measure the length of our chunks
        
        # initializing text splitter
        textSplitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap = 200,
            length_function = len
        )
        # generating chunks
        chunks = textSplitter.split_text(text)
            
        
        # Now, when user asks a question, langchain will look into the chunks to see which chunk contain info related to the question
        # After relevant chunks are gatered, those are fed to the language model to generate embeddings
        
        # Embedding are noting but vector representation (number representation) of the meaninf of your text.
        # Then thse, embedding are fed to our knowledge base - a document object
        
        # Generate embeddings and create vector store to search on
        
        # FAISS - to build knowledge base
        # FAISS (Facebook AI Similarity Search) is a library that allows developers to quickly search 
        # for embeddings of multimedia documents that are similar to each other. 
        # It solves limitations of traditional query search engines that are optimized for 
        # hash-based searches, and provides more scalable similarity search functions.
        
        
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks,embeddings)
        
        # When the question is asked by the user, it is embedded using the same 
        # embeding technique we used to embed our text chunks
        # This will allow us to perform a semantic search on the knowledge base
        
        # show user input
        
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # getting the chunks that contain info about the question
            docs = knowledge_base.similarity_search(user_question)
            
            # The search result will be a set of chunks and these chunks will be fed to the langchain model to generate an answer 
            # and reply back to user
            
            # creating the language model
            llm = OpenAI()
            # creating chain to string the response from llm
            chain = load_qa_chain(llm,chain_type="stuff")
            # running the chain to get an answer
            # we use get_openai_callback to get a callback from open once the chain is run to see the cost related details
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs,question = user_question)
                print(cb)
                
            st.write(response)
                
        
        
        
        
        
        
        
        
        
        
    
# To make sure application is being run directly and not being imported
if __name__ == '__main__':
    main()
    

    