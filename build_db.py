import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize the multilingual embedding model (LaBSE)
print("Loading AI model...")
model = SentenceTransformer('sentence-transformers/LaBSE')

# 2. Setup ChromaDB to save to your local folder
db_client = chromadb.PersistentClient(path="./vector_store")

# Forcefully delete the old database to prevent dimension errors (384 vs 768)
try:
    db_client.delete_collection(name="company_knowledge")
    print("Old database successfully deleted!")
except Exception:
    pass # If it's already gone, just ignore and move on

# Create a brand new, empty collection
collection = db_client.create_collection(name="company_knowledge")

# 3. Load your Excel files
print("Loading Excel files...")
df_en = pd.read_excel('english_qa.xlsx')
df_my = pd.read_excel('myanmar_qa.xlsx')

# Combine them into one dataframe
df_combined = pd.concat([df_en, df_my], ignore_index=True)

# 4. Generate Vectors and Store them
print("Generating vectors and saving to database...")
documents = []
embeddings = []
metadatas = []
ids = []

for index, row in df_combined.iterrows():
    # Using 'Questions' and 'Answers' to match your Excel columns perfectly
    question = str(row['Questions'])
    answer = str(row['Answers'])
    
    # Create the 768-dimension vector for the question
    vector = model.encode(question).tolist()
    
    documents.append(question)
    embeddings.append(vector)
    metadatas.append({"answer": answer})
    ids.append(f"qa_{index}")

# Save to ChromaDB
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("Database built successfully! You now have a fresh vector store.")