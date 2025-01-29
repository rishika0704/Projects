from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
from rag import RAGChatbot

# Define the path where uploaded PDFs will be stored
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
DB_PATH = os.path.join(os.getcwd(), 'chat_history.db')

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the RAG chatbot
chatbot = RAGChatbot()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
chatbot.initialize_model(api_key=GOOGLE_API_KEY)

# Database setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        role TEXT,
                        content TEXT
                    )''')
    conn.commit()
    conn.close()

init_db()

# Function to store chat messages in DB
def store_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO chat_history (timestamp, role, content) VALUES (?, ?, ?)",
                   (timestamp, role, content))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Handle PDF uploads."""
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(pdf_file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_file.save(temp_filepath)
        return jsonify({"message": "PDF uploaded successfully", "path": temp_filepath})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_chat_response', methods=['POST'])
def generate_chat_response():
    """Generate a chatbot response based on user input and uploaded PDF."""
    prompt = request.form['prompt']
    pdf_path = request.form['pdf_path']

    if not pdf_path:
        return jsonify({'error': 'No PDF path provided'}), 400

    try:
        # Store user query in the database
        store_message('user', prompt)
        
        # Process the PDF and create a retriever
        texts = chatbot.process_pdf(pdf_path)
        retriever = chatbot.create_retriever(texts, embeddings_model_name="models/embedding-001")

        # Generate the chatbot response
        response = chatbot.generate_response(prompt, retriever)
        
        # Store chatbot response in the database
        store_message('system', response)
        
        return jsonify({"chatbot_response": response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_chat_history():
    """Retrieve and return the chat history from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, role, content FROM chat_history ORDER BY timestamp ASC")
        history = cursor.fetchall()
        conn.close()
        return jsonify({"chat_history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
