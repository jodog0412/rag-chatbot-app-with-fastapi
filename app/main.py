from fastapi import FastAPI, File, UploadFile, Request, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain
from utils import load_split_pdf_file, build_history_aware_retriever, build_qa_chain
from dotenv import load_dotenv

load_dotenv()
templates = Jinja2Templates(directory="../templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
embedding = OpenAIEmbeddings()

@app.get("/", response_class=HTMLResponse)
def return_homepage(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/")
def upload_pdf_file(request: Request, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(f'../data/{file.filename}', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    if not file.filename.endswith('.pdf'):
        return {"message": "file format should be .pdf"}
    docs = load_split_pdf_file(f'../data/{file.filename}')
    db = Chroma.from_documents(persist_directory="../data",
                               documents=docs, 
                               embedding=embedding)
    return RedirectResponse(request=request, url='/chatting')

@app.get("/chatting", response_class=HTMLResponse)
def return_chatting_page(request: Request):
    return templates.TemplateResponse(request=request, name="chatting.html")

@app.websocket("/chatting")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        user_input = await websocket.receive_text()
        
        db = Chroma(persist_directory="../data", embedding_function=embedding)
        retriever = db.as_retriever()

        history_aware_retriever = build_history_aware_retriever(llm, retriever)
        qa_chain = build_qa_chain(llm)
        history_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        ### Statefully manage chat history ###
        store = {}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            history_rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": "default"}
            },  
        )["answer"]
        await websocket.send_text(response)