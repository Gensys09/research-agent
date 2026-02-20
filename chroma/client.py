import chromadb

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="memory")

collection.add(
    ids=["id1", "id2"],
    documents=[
        "Plastic is the most harmful pollutant",
        "Tenet is the best movie"
    ]
)

results = collection.query(
    query_texts=["This is a query document about contaminants"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)