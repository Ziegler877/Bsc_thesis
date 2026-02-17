import argparse
import os
import torch
import json
from tqdm import tqdm
import config
from src import data_loader
from src.rag_engine import AuthorRetriever, LLMGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Generator Model (llama2, llama4, scout)")
    parser.add_argument("--dataset", type=str, required=True, choices=["reuters", "darkreddit"])
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to .pt file (e.g., results/embeddings/e5_large_reuters.pt)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of candidates to retrieve (3 for Llama2, 5-10 for Scout)")
    parser.add_argument("--limit", type=int, default=100, help="Only run on first N test samples (for testing)")
    args = parser.parse_args()

    print(f"--- STARTING RAG PIPELINE ---")
    print(f"   Generator: {args.model}")
    print(f"   Retrieval: {args.embeddings}")
    print(f"   Candidates (K): {args.k}")

    # 1. Daten laden (Texte)
    print("   [Data] Loading raw texts...")
    train_txt, train_lbl = data_loader.load_dataset(args.dataset, "train")
    test_txt, test_lbl = data_loader.load_dataset(args.dataset, "test")

    # 2. Retriever initialisieren (Der Suchhund)
    retriever = AuthorRetriever(args.embeddings, train_txt, train_lbl)

    # Test-Embeddings aus der gleichen Datei laden (für die Suche)
    emb_data = torch.load(args.embeddings, map_location="cpu")
    test_vecs = emb_data['test_vecs']  # Die Vektoren der Test-Daten

    # 3. Generator initialisieren (Der Experte)
    generator = LLMGenerator(args.model)

    # 4. Main Loop
    results = []

    # Limitieren für Tests (Prompting dauert länger als Embeddings!)
    limit = min(args.limit, len(test_txt))
    print(f"   [Run] Processing {limit} test samples...")

    for i in tqdm(range(limit)):
        # A. Suche (Retrieval)
        # Wir nutzen den vorberechneten Vektor des Test-Samples
        query_vec = test_vecs[i]
        true_author = test_lbl[i]
        unknown_text = test_txt[i]

        # Hole Top-K Kandidaten
        candidates = retriever.retrieve(query_vec, k=args.k)

        # Check: Ist der richtige Autor überhaupt in den Kandidaten? (Recall@K)
        candidate_authors = [c['author'] for c in candidates]
        hit_in_retrieval = true_author in candidate_authors

        # B. Generierung (Prompting)
        response, prompt_used = generator.generate_decision(unknown_text, candidates)

        # C. Ergebnis speichern
        results.append({
            "id": i,
            "true_author": true_author,
            "candidates": candidate_authors,
            "hit_retrieval": hit_in_retrieval,  # War das Embedding gut genug?
            "llm_response": response,
            "prompt": prompt_used
        })

        # Zwischenspeichern alle 10 Schritte
        if i % 10 == 0:
            save_results(results, args.model, args.dataset)

    # Final speichern
    save_results(results, args.model, args.dataset)
    print("   [Done] Finished.")


def save_results(results, model, dataset):
    out_file = os.path.join(config.RAG_DIR, f"rag_results_{model}_{dataset}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()