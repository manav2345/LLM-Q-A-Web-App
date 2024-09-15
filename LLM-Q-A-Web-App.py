'''import os
os.system('pip install langchain')
os.system('pip install flask')
os.system('pip install bs4')
os.system('pip install sentence_transformers')
os.system('pip install groq')'''

from flask import Flask, request, render_template_string
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from groq import Groq


app = Flask(__name__)

# Load the pre-trained model once when the app starts
model = SentenceTransformer('all-MiniLM-L6-v2')

# HTML template as a string
html_template = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Colorful Form Example</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #333;
        }
        .container {
            background: #fff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            width: 400px;
            text-align: center;
            border: 2px solid #007bff;
        }
        h1 {
            margin-bottom: 30px;
            font-size: 32px;
            color: #007bff;
        }
        form {
            margin-bottom: 30px;
        }
        input[type="text"] {
            width: calc(100% - 24px);
            padding: 15px;
            margin: 12px 0;
            border: 2px solid #007bff;
            border-radius: 8px;
            box-sizing: border-box;
            font-size: 16px;
        }
        input[type="submit"], input[type="button"] {
            background-color: #28a745;
            color: #fff;
            border: none;
            padding: 15px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.3s ease;
            margin: 8px;
        }
        input[type="submit"]:hover, input[type="button"]:hover {
            background-color: #218838;
        }
        .clear-button {
            background-color: #dc3545;
        }
        .clear-button:hover {
            background-color: #c82333;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #eaf2f8;
            border: 2px solid #007bff;
            border-radius: 8px;
            text-align: left;
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
        .result h2 {
            margin-top: 0;
            color: #007bff;
        }
        .result p {
            margin: 10px 0;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Submit Your Data</h1>
        <form method="post">
            URL: <input type="text" name="url" value="{{ url }}" placeholder="Enter URL"><br>
            Question: <input type="text" name="question" value="{{ question }}" placeholder="Enter Question"><br>
            <input type="submit" value="Process URL">
            <input type="button" class="clear-button" value="Clear All" onclick="clearForm()">
        </form>
        <div class="result">
            <h2>Answer:</h2>
            <p>{{ answer }}</p>
        </div>
    </div>
    <script>
        function clearForm() {
            document.querySelector('input[name="url"]').value = '';
            document.querySelector('input[name="question"]').value = '';
            // Optionally, clear the result section
            document.querySelector('.result p').innerHTML = '';
        }
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    url = ''
    question = ''
    answer = ''
    
    if request.method == 'POST':
        # Get URL and question from user input
        url = request.form.get('url', '')
        question = request.form.get('question', '')
        
        if url and question:
            try:
                # Step 1: Scrape the website content
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                
                # Step 2: Split the text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                
                # Step 3: Embed the chunks and the user query
                chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
                query_embedding = model.encode(question, convert_to_tensor=True)
                
                # Step 4: Find the most similar chunk using cosine similarity
                similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)
                most_similar_idx = similarities.argmax()
                most_similar_chunk = chunks[most_similar_idx]
                
                # Step 5: Use Groq API to generate a response based on the most relevant chunk
                client = Groq(api_key='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')
                completion = client.chat.completions.create(
                    model="llama3-groq-8b-8192-tool-use-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": f"{question}? Answer based on the paragraph: {most_similar_chunk}"
                        }
                    ],
                    temperature=0.5,
                    max_tokens=1024,
                    top_p=0.65,
                    stream=False,
                )
                
                # Extract and display the generated response
                answer = completion.choices[0].message.content

            except Exception as e:
                answer = f"Error: {str(e)}"
    
    return render_template_string(html_template, url=url, question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
