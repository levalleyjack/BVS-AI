from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'r') as f:
        text = " ".join([page.strip("") for page in f])
    return text

def split_text_into_chunks(text, chunk_size=512):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_bvs(bvs_path):
    with open(bvs_path, 'r') as f:
        return json.load(f)

def get_relevant_embeddings(pdf_path, bvs_data, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    
    chunk_embeddings = model.encode(chunks)    
    results = []
    
    for category, statements in bvs_data.items():
        bvs_embeddings = model.encode(statements)
        
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarities = cosine_similarity([chunk_embedding], bvs_embeddings)[0]
            best_match_idx = similarities.argmax()
            best_match_score = similarities[best_match_idx]
            best_match_statement = statements[best_match_idx]
            if best_match_score > 0.55: 
                results.append({
                    "category": category,
                    "statement": best_match_statement,
                    "chunk": chunks[i],
                    "similarity_score": best_match_score
                })
    
    return results

pdf_path = "/pdfs/10q.txt" 
bvs_path = "BVS.json"

bvs_data = load_bvs(bvs_path)

results = get_relevant_embeddings(pdf_path, bvs_data)

for result in results:
    print(f"Category: {result['category']}")
    print(f"Related BVS: {result['statement']}")
    print(f"Extracted Chunk: {result['chunk']}")
    print(f"Similarity Score: {result['similarity_score']:.4f}")
    print("\n")
