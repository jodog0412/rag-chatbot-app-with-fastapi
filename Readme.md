# fastapi-app-with-langchain
This project is purposed to create RAG(retriever-augmented-generation) chatbot application using fastapi and docker.

If you upload `.pdf` file, RAG chatbot performs Q&A based on your document.

This project heavily referenced @AshishSinha5’s rag_api project.

<img src="https://github.com/user-attachments/assets/e502f8cb-a38f-463d-b918-3a445f3851f8" width=800 height=450>

## Table of Content
- Getting Started
- Usage
    - Starting the local server
    - Deploying the server
## Getting Started
1. Clone this repository to your local machine.
    
    ```powershell
    git clone https://github.com/jodog0412/langchain-app-with-fastapi.git
    cd fastapi-app-with-langchain
    ```
    
2. Create a virtual environment in your local directory and activate it.
    
    ```powershell
    python -m venv .venv
    .venv/bin/activate.bat
    ```
    
3. Install Python packages in your virtual environment.
    
    ```
    pip install -r requirements.txt
    ```
    
4. Set `OPEN_AI_API_KEY` on `.env` file in the `app` directory.
    
    ```
    # app/.env
    OPENAI_API_KEY =
    ```
# Usage
## Starting the local server

```powershell
cd app
uvicorn main:app --reload
```

## Deploying the server

1. Set `OPEN_AI_API_KEY` on `Dockerfile`.
    
    ```Docker
    FROM python:3.10
    
    WORKDIR /code
    
    COPY . /code
    
    RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
    
    ENV OPENAI_API_KEY "ENTER YOUR OPENAI API KEY"
    
    WORKDIR /code/app
    
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    ```
    
2. Build docker-image.
    
    ```powershell
    docker build --pull --rm -f "Dockerfile" -t langchainappwithfastapi:latest "." 
    ```
    
3. Create docker-container.
    
    ```powershell
    docker run -d --name container -p 8000:8000 langchainappwithfastapi
    ```
# Features/Update
- [x]  build history-aware RAG chatbot
- [x]  implement client-server RESTAPI
- [x]  implement websocket connection
- [x]  deploy the app
- [ ]  bug fixes
- [ ]  support local LLM(only supports `GPT` model in current version)
- [ ]  implement RESTful-API
