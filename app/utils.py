from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def load_split_pdf_file(pdf_file: Annotated[any, "file format should be .pdf"]):
    loader = PyPDFLoader(pdf_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    data = loader.load_and_split(text_splitter)
    return data

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)