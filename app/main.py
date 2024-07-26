from fastapi import FastAPI, File, UploadFile
from load_data import load_split_pdf_file
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
llm = ChatOpenAI(model = "gpt-4o-mini")
embedding = OpenAIEmbeddings()

@app.post("/uploadfile/")
async def upload_pdf_file(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(f'../data/{file.filename}', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    assert file.filename.endswith('.pdf'), 'file format sholud be .pdf'
    docs = load_split_pdf_file(f'../data/{file.filename}')
    db = Chroma.from_documents(persist_directory="../data",
                               documents=docs, 
                               embedding=embedding)
    return {"message" : f"Successfully uploaded {file.filename}"}

@app.get("/llm_respond")
def llm_respond():
    db = Chroma(persist_directory="../data", embedding_function=embedding)
    retriever = db.as_retriever()
    prompt =  hub.pull("rlm/rag-prompt")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke("이력서에서 발견되는 지원자의 강점을 자세히, 차근차근 설명해줘.")
    return {"response" : response}
