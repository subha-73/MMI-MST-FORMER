import os
import re
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from config import TEXT_REPORTS_DIR, PATIENT_DATA_PATH

# Path to stored guideline chunks
DOC_FILE = "C:/Harini/LLM/database/faiss_db/documents.pkl"

# Load a stronger embedding model
model = SentenceTransformer("all-mpnet-base-v2")  # higher quality than MiniLM

# -------------------- Helper functions --------------------

# Cache embeddings to avoid recomputation
embedding_cache = {}

def get_embedding(text):
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    return embedding_cache[text]

def clean_report(text):
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'-{3,}', '', text)
    text = re.sub(r'^\s*-\s*', '', text, flags=re.MULTILINE)
    text = re.sub(
        r'^(Generated|Patient ID|Current MD|Progression Rate|VFI|PSD|Mean VF|Severity|Clinical Summary|CLINICAL REPORT|Report Generated|Analysis Period).*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def split_sentences(text):
    text = clean_report(text)
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [
        s.strip() for s in raw
        if len(s.strip()) > 25
        and any(len(w) > 4 for w in s.split())
        and not s.strip().startswith('#')
        and not s.strip().startswith('*')
    ]

def similarity(a, b):
    return float(util.cos_sim(get_embedding(a), get_embedding(b)))

# -------------------- Evaluation Metrics --------------------

# -------------------- Updated Metrics --------------------

def context_precision(retrieved_chunks, patient_data, threshold=0.5):
    query = (
        f"{patient_data.get('severity', '')} glaucoma patient, "
        f"MD {patient_data.get('MD')} dB, "
        f"VFI {patient_data.get('VFI')} percent, "
        f"Eye: {patient_data.get('eye')}, "
        f"visual field progression management guidelines"
    )
    query_embedding = get_embedding(query)

    # Filter chunks mentioning patient eye, severity, MD, or VFI
    patient_chunks = [
        c for c in retrieved_chunks
        if (patient_data.get('eye', '').lower() in c.lower()) or
           (patient_data.get('severity', '').lower() in c.lower()) or
           (str(patient_data.get('MD')) in c) or
           (str(patient_data.get('VFI')) in c)
    ]

    if not patient_chunks:
        return 0
    relevant = sum(1 for c in patient_chunks if float(util.cos_sim(get_embedding(c), query_embedding)) > threshold)
    return relevant / len(patient_chunks)


def context_recall(retrieved_chunks, patient_data, threshold=0.5):
    query = (
        f"{patient_data.get('severity', '')} glaucoma patient, "
        f"MD {patient_data.get('MD')} dB, "
        f"VFI {patient_data.get('VFI')} percent, "
        f"Eye: {patient_data.get('eye')}, "
        f"visual field progression management guidelines"
    )
    query_embedding = get_embedding(query)

    patient_chunks = [
        c for c in retrieved_chunks
        if (patient_data.get('eye', '').lower() in c.lower()) or
           (patient_data.get('severity', '').lower() in c.lower()) or
           (str(patient_data.get('MD')) in c) or
           (str(patient_data.get('VFI')) in c)
    ]

    if not patient_chunks:
        return 0
    relevant = sum(1 for c in patient_chunks if float(util.cos_sim(get_embedding(c), query_embedding)) > threshold)
    total_relevant = len(patient_chunks)
    return relevant / total_relevant if total_relevant else 0


def faithfulness(report, retrieved_chunks, threshold=0.5):
    """
    Fraction of report sentences supported by retrieved chunks above threshold
    """
    sentences = split_sentences(report)
    if not sentences:
        return 0
    supported = 0
    for s in sentences:
        s_embed = get_embedding(s)
        # check if any chunk supports this sentence above threshold
        if any(float(util.cos_sim(s_embed, get_embedding(c))) > threshold for c in retrieved_chunks):
            supported += 1
    return supported / len(sentences)


def answer_relevancy(report, patient_data, threshold=0.5):
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
    with open(DOC_FILE, "rb") as f:
        documents = pickle.load(f)

    with open(PATIENT_DATA_PATH, "r") as f:
        patients = json.load(f)

    report_files = sorted([f for f in os.listdir(TEXT_REPORTS_DIR) if f.endswith(".txt")])
    results = []

    # Split documents into paragraphs
    retrieved_chunks = []
    for doc in documents[:100]:
        text = doc.get("text", "") if isinstance(doc, dict) else getattr(doc, "page_content", str(doc))
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
        retrieved_chunks.extend(paragraphs)

    for i, file in enumerate(report_files):
        report_path = os.path.join(TEXT_REPORTS_DIR, file)
        with open(report_path, "r", encoding="utf-8") as f:
            report = f.read()

        patient = patients[i] if i < len(patients) else {}

        precision = context_precision(retrieved_chunks, patient)
        recall = context_recall(retrieved_chunks, patient)
        faithful = faithfulness(report, retrieved_chunks)
        relevancy = answer_relevancy(report, patient)

        results.append({
            "report": file,
            "context_precision": round(precision, 3),
            "context_recall": round(recall, 3),
            "faithfulness": round(faithful, 3),
            "answer_relevancy": round(relevancy, 3)
        })

    print("\nRAG Evaluation Results\n")
    for r in results:
        print(r)

    print("\nAverage RAG Metrics\n")
    print("Context Precision:", round(np.mean([r["context_precision"] for r in results]), 3))
    print("Context Recall:",    round(np.mean([r["context_recall"] for r in results]), 3))
    print("Faithfulness:",      round(np.mean([r["faithfulness"] for r in results]), 3))
    print("Answer Relevancy:",  round(np.mean([r["answer_relevancy"] for r in results]), 3))

if __name__ == "__main__":
    run_evaluation()