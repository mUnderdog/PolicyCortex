from retriever import retrieve_docs

query = "How should startups secure APIs?"

results = retrieve_docs(query)

for r in results:
    print("\nCONTROL:", r["control"])
    print("SOURCE:", r["source"])
    print("CONTENT:", r["content"])