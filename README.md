# Assignment 1: Sentiment Analysis

```
In this there are:
- Data Folder : This has the data for IMDB Movie 
- Reviews and the data is from kaggle. 
- Baseline Model : This ipynb file contains the code from database setup to baseline (logistic model) 
                   creation. 
- imdb.db : It is the sqlite database. 
- logistic_model.pkl : Pickle file for logistic regression model. 
- app.py : Flask api for the prediction
- Sentiment Analysis - Bert: This is ipynb file for bert classification, I have not uploaded the 
                             models since the files were quite huge and github didn't support.
                             The whole code in this file is on GPU, because CPU was taking very long time (used Google Colab)
- vectorizer.pkl : Pickle file for vectors.
-requirements.txt : All the requirements 

```

### Setup

Clone the repository 
```bash
git clone https://github.com/rishika0704/Projects.git
```
Go to Assignment 1 folder
```bash
cd "Assignment 1"
```
Create a virtual environment 
```bash
python -m venv assignment1
```
Activate the venv
(The below command is for cmd)
```bash
assignment1\Scripts\activate
```
Install all the required libraries
```bash
pip install -r requirements.txt
```
### Running the flask server
```bash
python app.py
```
### Testing using the terminal 
Below is for bash
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"text": "The movie is too overrated..!!"}'
```

# Assignment 2: RAG
```
In this there are:
- templates : Html files
- uploads : Folder for storing the pdf's user uploads 
            (this has a pdf file for testing the RAG in app.py)
- app.py : Flask api for running the RAG (here, i am using chroma and gemini api)
- chat_history.db - sqlite db for storing the chat history
- rag.py - Contains the RAG model
- RAG - ollama & pinecone.ipynb : This is a ipynb file where i am using ollama and pinecone 
        (vector database) for creating a RAG model. The code is running on GPU so used google colab. 
- Recipes.py : This is the pdf file which i used for creating a RAG model in 
               "RAG - ollama & pinecone.ipynb". The data is getting stored in 
               pinecone vector database.
-requirements.txt : All the requirements 

```
### Setup

Clone the repository 
```bash
git clone https://github.com/rishika0704/Projects.git
```
Go to Assignment 1 folder
```bash
cd "Assignment 2"
```
Create a virtual environment 
```bash
python -m venv assignment2
```
Activate the venv
(The below command is for cmd)
```bash
assignment2\Scripts\activate
```
Install all the required libraries
```bash
pip install -r requirements.txt
```
### Running the flask server
```bash
python app.py
```
### Testing using the terminal 
Below is for bash
#### Uploading the file
```bash
curl -X POST http://127.0.0.1:5000/upload_pdf \
-H "Content-Type: multipart/form-data" \
-F "pdf_file=@/path/to/your/file.pdf"
```
#### Generating Chat Response
```bash 
curl -X POST http://127.0.0.1:5000/generate_chat_response \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "prompt=What is neurons?&pdf_path=/path/to/uploaded/file.pdf"
```
#### Retrieve Chat History 
```bash
curl -X GET http://127.0.0.1:5000/history
```
You can go directly to the http://127.0.0.1:5000 and test after running app.py
