import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os
import sys

# Pfad-Hack, um config zu finden
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AuthorRetriever:
    """
    Der 'Suchhund': Findet relevante Kandidaten basierend auf Embeddings.
    """

    def __init__(self, embedding_path, train_texts, train_labels):
        print(f"   [Retriever] Loading embeddings from {embedding_path}...")
        data = torch.load(embedding_path, map_location="cpu")

        # Wir brauchen die Trainings-Vektoren, um darin zu suchen
        self.db_vecs = data['train_vecs']  # Shape: [N_samples, Hidden_dim]
        self.db_labels = data['train_labels']
        self.db_texts = train_texts  # Wir brauchen die Originaltexte für den Prompt!

        # Normalisierung für Cosine Similarity
        self.db_vecs = F.normalize(self.db_vecs, p=2, dim=1)

    def retrieve(self, query_vec, k=3):
        """
        Findet die Top-K ähnlichsten Texte aus der Datenbank.
        Gibt zurück: Liste von (Autor, Text) Tupeln.
        """
        # query_vec muss [1, Dim] sein
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)

        query_vec = F.normalize(query_vec, p=2, dim=1)

        # Cosine Similarity: Dot Product
        scores = torch.mm(query_vec, self.db_vecs.t())  # [1, N_samples]

        # Top K
        topk_scores, topk_indices = torch.topk(scores, k=k)

        results = []
        found_authors = set()

        # Wir wollen K *verschiedene* Autoren (optional, hier nehmen wir einfach die besten Samples)
        for idx in topk_indices[0]:
            idx = idx.item()
            author = self.db_labels[idx]
            text = self.db_texts[idx]
            results.append({"author": author, "text": text})

        return results


class LLMGenerator:
    """
    Der 'Experte': Llama 2 oder Scout für Textgenerierung.
    """

    def __init__(self, model_alias):
        self.model_alias = model_alias
        self.device = config.DEVICE

        # Modellpfad bestimmen
        if "scout" in model_alias:
            model_path = config.LLAMA4_SCOUT_CHECKPOINT_DIR
        elif "llama4" in model_alias:  # Maverick
            model_path = config.LLAMA4_CHECKPOINT_DIR
        elif "llama2" in model_alias:
            model_path = config.LLAMA2_CHECKPOINT_DIR
        else:
            raise ValueError(f"Unknown model: {model_alias}")

        print(f"   [Generator] Loading {model_alias} from {model_path} (4-bit)...")

        # 4-Bit Config (Zwingend für Maverick/Scout auf A40/A100)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def generate_decision(self, unknown_text, candidates):
        """
        Baut den Prompt und holt die Antwort.
        """
        # 1. Prompt Bauen (Dynamic Few-Shot)
        prompt = self._construct_prompt(unknown_text, candidates)

        # 2. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 3. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # Antwortlänge begrenzen
                temperature=0.1,  # Geringe Kreativität für präzise Antworten
                do_sample=True
            )

        # 4. Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Nur den neuen Text (die Antwort) extrahieren
        # (Llama wiederholt oft den Prompt, wir schneiden ihn ab)
        response_only = full_response[len(prompt):].strip()

        return response_only, prompt

    def _construct_prompt(self, unknown_text, candidates):
        """
        Erstellt den 'Forensic Linguist' Prompt.
        """
        candidate_text = ""
        candidate_list = []

        for i, cand in enumerate(candidates):
            c_id = f"Candidate {chr(65 + i)}"  # A, B, C...
            candidate_list.append(f"{c_id} ({cand['author']})")  # Merken wir uns intern

            # Text kürzen, damit er in den Context passt (ca 150 Wörter)
            snippet = " ".join(cand['text'].split()[:150])
            candidate_text += f"\n--- {c_id} ---\n{snippet}...\n"

        # Der unbekannte Text (auch gekürzt)
        unknown_snippet = " ".join(unknown_text.split()[:200])

        prompt = f"""[INST] You are an expert Forensic Linguist. Your task is to perform authorship attribution.

I will provide you with an Unknown Text and samples from {len(candidates)} Candidate Authors.
Analyze the stylistic features (syntax, vocabulary, punctuation) to identify the true author.

=== UNKNOWN TEXT ===
{unknown_snippet}...

=== CANDIDATES ===
{candidate_text}

=== TASK ===
1. Analyze the writing style of the Unknown Text.
2. Compare it to the Candidates.
3. Identify which Candidate (A, B, C...) wrote the text.

Answer format: "Reasoning: [Brief explanation]. Final Answer: Candidate [X]"
[/INST]
Reasoning:"""
        return prompt