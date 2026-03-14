import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from config import TEXT_REPORTS_DIR, PATIENT_DATA_PATH

# Path to stored guideline chunks
DOC_FILE = "C:/Harini/LLM/database/faiss_db/documents.pkl"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------- Helper functions --------------------

# Cache embeddings to avoid recomputation
embedding_cache = {}

def get_embedding(text):
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    return embedding_cache[text]

def split_sentences(text):
    return [s.strip() for s in text.replace("\n", ".").split(".") if len(s.strip()) > 10]

def similarity(a, b):
    return float(util.cos_sim(get_embedding(a), get_embedding(b)))

# -------------------- Evaluation Metrics --------------------

def context_precision(retrieved_chunks,patient_data, threshold=0.25):
    query = f"""
glaucoma patient
MD {patient_data.get('MD')}
VFI {patient_data.get('VFI')}
Severity {patient_data.get('severity')}
Eye {patient_data.get('eye')}
Glaucoma progression management guidelines
"""
    if not retrieved_chunks: 
        return 0
    relevant = sum(1 for chunk in retrieved_chunks if similarity(chunk, query) > threshold)
    return relevant / len(retrieved_chunks)

def context_recall(retrieved_chunks, threshold=0.35):
    query = "glaucoma visual field progression management guidelines"
    if not retrieved_chunks: 
        return 0
    relevant = sum(1 for chunk in retrieved_chunks if similarity(chunk, query) > threshold)
    total_relevant = max(len(retrieved_chunks), 1)
    return relevant / total_relevant

def faithfulness(report, retrieved_chunks, threshold=0.3):
    sentences = split_sentences(report)
    if not sentences:
        return 0
    supported = sum(1 for s in sentences if any(similarity(s, c) > threshold for c in retrieved_chunks))
    return supported / len(sentences)

def answer_relevancy(report, patient_data, threshold=0.25):
    
    severity = patient_data.get('severity', 'moderate')
    md = patient_data.get('MD', '')
    vfi = patient_data.get('VFI', '')
    eye = patient_data.get('eye', '')
    prog_rate = patient_data.get('progression_rate', '')

    queries = [
        f"glaucoma {severity} severity MD {md} dB mean deviation visual field",
        f"VFI {vfi} visual field index progression glaucoma management",
        f"glaucoma treatment IOP target intraocular pressure control",
        f"visual field loss progression rate {prog_rate} monitoring follow up",
        f"glaucoma management recommendations urgent review preserve vision"
    ]

    sentences = split_sentences(report)
    if not sentences:
        return 0

    relevant = 0
    for sentence in sentences:
        max_sim = max(similarity(sentence, q) for q in queries)
        if max_sim > threshold:
            relevant += 1

    return relevant / len(sentences)

# -------------------- Main Evaluation --------------------

def run_evaluation():
    # Load guideline chunks (precomputed)
    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)

    # Load patient data
    with open(PATIENT_DATA_PATH, "r") as f:
        patients = json.load(f)

    # Prepare report files
    report_files = sorted([f for f in os.listdir(TEXT_REPORTS_DIR) if f.endswith(".txt")])
    results = []

    # Precompute embeddings for all chunks once
    retrieved_chunks = []
    for doc in documents[:100]:
        if isinstance(doc, dict):
            retrieved_chunks.append(doc.get("text", ""))
        elif hasattr(doc, "page_content"):
            retrieved_chunks.append(doc.page_content)
        else:
            retrieved_chunks.append(str(doc))
    # Embeddings cached now

    for i, file in enumerate(report_files):
        report_path = os.path.join(TEXT_REPORTS_DIR, file)
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

        patient = patients[i] if i < len(patients) else {}

        # Compute metrics
        precision = context_precision(retrieved_chunks,patient)
        recall = context_recall(retrieved_chunks)
        faithful = faithfulness(report, retrieved_chunks)
        relevancy = answer_relevancy(report, patient)

        results.append({
            "report": file,
            "context_precision": round(precision, 3),
            "context_recall": round(recall, 3),
            "faithfulness": round(faithful, 3),
            "answer_relevancy": round(relevancy, 3)
        })

    # Display results
    print("\nRAG Evaluation Results\n")
    for r in results:
        print(r)

    print("\nAverage RAG Metrics\n")
    avg_precision = np.mean([r["context_precision"] for r in results])
    avg_recall = np.mean([r["context_recall"] for r in results])
    avg_faithfulness = np.mean([r["faithfulness"] for r in results])
    avg_relevancy = np.mean([r["answer_relevancy"] for r in results])

    print("Context Precision:", round(avg_precision, 3))
    print("Context Recall:", round(avg_recall, 3))
    print("Faithfulness:", round(avg_faithfulness, 3))
    print("Answer Relevancy:", round(avg_relevancy, 3))

if __name__ == "__main__":
    run_evaluation()