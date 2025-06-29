import os
import re
import ast
import uuid
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from flask import render_template

# =============== SETUP ===============

app = Flask(__name__)

# Gemini setup
genai.configure(api_key="AIzaSyBAokGJ91nfl-UG5cGm3c3tzvwrdGBxYqc")
model_llm = genai.GenerativeModel("gemini-1.5-flash")

# Embedder
embedder = SentenceTransformer("intfloat/e5-base-v2")

# Pinecone client
pc = Pinecone(api_key="pcsk_4QBiy4_8zZzjsRknsPZBupzZABdbGsE2EFXLiSkFnPhKz9P9Vt8Uzd8V39znyGkmFDksfT")
index_name = "mosfet-datasheet-2"

# Create index if not exists
existing_indexes = [idx.name for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=768,  # 👈 MATCH your model dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# =============== HELPERS ===============

def row_to_text(row):
    return (
        f"Datasheet: {row['Datasheet']}\n"
        f"Image: {row['Image']}\n"
        f"DK Part #: {row['DK Part #']}\n"
        f"Mfr Part #: {row['Mfr Part #']}\n"
        f"Mfr: {row['Mfr']}\n"
        f"Supplier: {row['Supplier']}\n"
        f"Description: {row['Description']}\n"
        f"Drain to Source Voltage (Vdss): {row['Drain to Source Voltage (Vdss)']}V\n"
        f"Rds On (Max) @ Id, Vgs: {row['Rds On (Max) @ Id, Vgs']}\n"
        f"Grade: {row['Grade']}\n"
        f"Package / Case: {row['Package /Case']}"
    )

def extract_rds_mohm(value):
    if pd.isna(value):
        return None
    value = str(value)
    match = re.search(r'([\d.]+)\s*(m?Ohm)', value, re.IGNORECASE)
    if match:
        number = float(match.group(1))
        unit = match.group(2).lower()
        return number if 'mohm' in unit else number * 1000
    return None

def clean_vdss_column(df):
    df["Drain to Source Voltage (Vdss)"] = (
        df["Drain to Source Voltage (Vdss)"]
        .astype(str)
        .str.replace("V", "", regex=False)
        .str.strip()
        .astype(float)
    )
    return df

# =============== ROUTES ===============

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_csv():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No CSV file uploaded"}), 400

    df = pd.read_csv(file)
    df = df.head(15)

    # Select needed columns
    columns = [
        "Datasheet", "Image", "DK Part #", "Mfr Part #", "Mfr", "Supplier",
        "Description", "Drain to Source Voltage (Vdss)", "Rds On (Max) @ Id, Vgs",
        "Grade", "Package /Case"
    ]
    filtered_df = df[columns]

    filtered_df = clean_vdss_column(filtered_df) #cleaning Vdss coloumn
    filtered_df["Rds On (Max) @ Id, Vgs"] = filtered_df["Rds On (Max) @ Id, Vgs"].apply(extract_rds_mohm)  #Cleaning RDS column

    # Format rows
    texts = filtered_df.apply(row_to_text, axis=1).tolist()

    # Embed
    embeddings = embedder.encode(texts).tolist()

    # Upsert
    for i, (text, vector) in enumerate(zip(texts, embeddings)): # Convert each row to a dict and ensure all values are native Python types
        row = filtered_df.iloc[i]
        metadata = {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in row.items()
        }

        # Upsert the vector with metadata
        index.upsert([(f"id-{i}", vector, metadata)])

    return jsonify({"message": f"Uploaded & embedded {len(texts)} rows"}), 200

# =============== RUN ===============

# --- Your embedding model & Gemini LLM ---
# model = your embedding model
# model_llm = your Gemini model

@app.route('/query', methods=['POST'])
def query_mosfet():
    data = request.get_json()
    user_query = data.get('query')

    # Step 1: Embed user query
    query_embedding = embedder.encode([user_query])[0]

    # Step 2: Use Gemini to extract numeric filter
    filter_gen = model_llm.generate_content(f"""
    Given the user query: "{user_query}", extract only numerical filter conditions related to:
    - "Drain to Source Voltage (Vdss)"
    - "Rds On (Max) @ Id, Vgs"

    Instructions:
    - Assume all Rds values in the data are in milliohms (mΩ).
    - If the query uses units like "milli Ohm", "mOhm", or "milliohm", use the number directly.
    - If the query uses units like "Ohm", multiply by 1000 to convert to milliohms.
    - Use operators like "$gt", "$lt", "$eq" to reflect the query intent.
    - Only include a key if it is clearly mentioned in the query.
    - Return ONLY a valid Python dictionary using the following format, and nothing else:

    Example:
    {{
      "Drain to Source Voltage (Vdss)": {{"$gt": 30}},
      "Rds On (Max) @ Id, Vgs": {{"$lt": 500}}
    }}
    """)

    raw_filter = filter_gen.text.strip()
    raw_filter = re.sub(r"^```[\w]*", "", raw_filter)
    raw_filter = re.sub(r"```$", "", raw_filter.strip())
    filters = ast.literal_eval(raw_filter)

    # Step 3: Query Pinecone
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        filter=filters
    )
    
    # Step 4: Convert matches to DataFrame and generate LLM response
    matched_mosfet_data = [m['metadata'] for m in results['matches']]
    if matched_mosfet_data:
        results_df = pd.DataFrame(matched_mosfet_data)
        desired_columns = ['Datasheet', 'Image', 'DK Part #', 'Mfr Part #', 'Mfr', 'Supplier', 'Description',
                         'Product Status', 'Drain to Source Voltage (Vdss)', 'Rds On (Max) @ Id, Vgs',
                         'Grade', 'Package /Case']
        existing_columns = [col for col in desired_columns if col in results_df.columns]
        results_df = results_df[existing_columns]

        # Generate LLM response with context
        contexts = "\n".join([str(row) for _, row in results_df.iterrows()])
        prompt = f"""Based on these MOSFET specifications:
        {contexts}

        Answer this question: {user_query}

        Provide:
        1. A concise result, only relevant rows from the given table. 
        2. In Table format, with most relevant columns. 
       """

        llm_response = model_llm.generate_content(prompt)

        # Return both structured data and LLM analysis
        return jsonify({
            "structured_data": results_df.to_dict(orient="records"),
            "analysis": llm_response.text
        })

    else:
        return jsonify({"message": "No matching MOSFETs found."})



if __name__ == "__main__":
    app.run(debug=True)
