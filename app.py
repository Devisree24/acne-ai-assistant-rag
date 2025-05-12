from flask import Flask, render_template, request
import openai
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Flask App
app = Flask(__name__)
load_dotenv()

# Setup OpenAI (via OpenRouter)
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Load and process documents
DOCUMENTS_DIR = "documents"
DOCUMENT_FILES = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
documents = []
doc_chunks = []

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())

# Read PDFs and chunk them
for file in DOCUMENT_FILES:
    full_path = os.path.join(DOCUMENTS_DIR, file)
    if os.path.exists(full_path):
        text = extract_text_from_pdf(full_path)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        doc_chunks.extend(chunks)

# Create embeddings for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(doc_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

@app.route('/', methods=['GET', 'POST'])
def query():
    result = ""
    query_text = ""

    if request.method == 'POST':
        query_text = request.form['query']
        if not query_text:
            result = "Please enter your main concern."
        else:
            query_vec = model.encode([query_text])
            _, I = index.search(np.array(query_vec), k=3)
            retrieved = [doc_chunks[i] for i in I[0]]

            prompt = (
                "You are a medical assistant AI specializing in acne.\n\n"
                "Task:\n"
                "1. Read the patient's main query carefully.\n"
                "2. Read the extracted document knowledge carefully.\n"
                "3. Answer the query using ONLY the provided documents. "
                "If you cannot find enough information, politely say so.\n\n"
                f"User Query:\n{query_text}\n\n"
                "Relevant Extracted Documents:\n"
                + "\n\n---\n\n".join(retrieved)
            )

            try:
                messages = [
                    {"role": "system", "content": "You are a medically accurate assistant focused on acne research and clinical best practices."},
                    {"role": "user", "content": prompt}
                ]

                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=messages,
                    temperature=0.4,
                    top_p=0.9
                )
                result = response.choices[0].message.content

            except Exception as e:
                result = f"Error occurred: {e}"

    return render_template("index.html", result=result, query=query_text)

if __name__ == '__main__':
    app.run(debug=True)
