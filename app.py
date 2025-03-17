from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import system_prompt
import google.generativeai as genai
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Gemini API Key setup
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Setup Pinecone retriever (Optional: can be replaced with Chroma/FAISS)
embeddings = download_hugging_face_embeddings()

index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    # Get user message from form
    msg = request.form["msg"]
    print("User input:", msg)

    # Retrieve relevant documents from Pinecone (optional RAG)
    docs = retriever.invoke(msg)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Format prompt for Gemini 1.5 Flash
    prompt = system_prompt.format(context=context) + f"\nQuestion: {msg}\nAnswer:"

    # Generate response from Gemini
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    # Get the response text
    result = response.text
    print("Gemini Response:", result)

    return result

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
