#!/usr/bin/env python3
"""
evaluate.py — Script d'évaluation de performance du chatbot Sonatrach.

Modes d'utilisation :
  1. Direct (sans serveur) :
       python evaluate.py --dataset test_dataset.json

  2. Via API (serveur en cours d'exécution) :
       python evaluate.py --dataset test_dataset.json --api-url http://localhost:8001 \
                          --email admin@example.com --password secret

  3. Avec juge LLM pour plus de précision :
       python evaluate.py --dataset test_dataset.json --llm-judge

Options :
  --output rapport.json    Sauvegarder le rapport en JSON
  --category structured    Tester seulement une catégorie (structured|rag|mixed)
  --verbose                Afficher les réponses complètes
"""

import sys, os, json, time, argparse, re, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import types
if "langchain" not in sys.modules:
    _lc_stub = types.ModuleType("langchain")
    _lc_stub.debug = False
    sys.modules["langchain"] = _lc_stub

# ─── ANSI couleurs ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _c(text, color): return f"{color}{text}{RESET}"


# ─── Métriques ────────────────────────────────────────────────────────────────

import unicodedata, re

def normalize(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Fraction des mots-clés attendus présents dans la réponse."""
    if not keywords:
        return 1.0
    norm_answer = normalize(answer)
    found = sum(1 for kw in keywords if normalize(kw) in norm_answer)
    return found / len(keywords)


def containment_score(answer: str, expected: str | None) -> float:
    """1.0 si la réponse contient exactement l'attendu, sinon 0."""
    if not expected:
        return None
    return 1.0 if normalize(expected) in normalize(answer) else 0.0


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """ROUGE-L simplifié (LCS / len(reference))."""
    if not reference or not hypothesis:
        return 0.0
    h = normalize(hypothesis).split()
    r = normalize(reference).split()
    if not h or not r:
        return 0.0
    m, n = len(h), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if h[i-1] == r[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / m if m else 0
    recall    = lcs / n if n else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def semantic_score(answer: str, reference: str, model) -> float:
    """Similarité cosine entre les embeddings de la réponse et de la référence."""
    if not reference or not answer:
        return None
    try:
        import numpy as np
        embs = model.encode([answer, reference], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
    except Exception:
        return None


def llm_judge_score(question: str, answer: str, llm, tokenizer) -> float:
    """
    Utilise le LLM local pour juger si la réponse est correcte.
    Retourne un score 0.0, 0.5, ou 1.0.
    """
    prompt = f"""Tu es un juge impartial. Évalue si la réponse du chatbot répond correctement à la question.

Question : {question}
Réponse du chatbot : {answer}

Réponds UNIQUEMENT par l'un de ces mots : CORRECT, PARTIEL, INCORRECT
- CORRECT : la réponse est juste et complète
- PARTIEL : la réponse est partiellement juste ou incomplète
- INCORRECT : la réponse est fausse ou hors sujet

Verdict :"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if hasattr(llm, "device"):
            inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        out = llm.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        verdict = decoded.strip().upper().split()[0] if decoded.strip() else ""
        if "CORRECT" in verdict and "IN" not in verdict:
            return 1.0
        elif "PARTIEL" in verdict:
            return 0.5
        else:
            return 0.0
    except Exception as e:
        print(f"    {_c('⚠ LLM judge échoué:', YELLOW)} {e}")
        return None


# ─── Pipeline direct ──────────────────────────────────────────────────────────

def load_pipeline():
    print(f"{_c('→', BLUE)} Chargement du pipeline RAG...")
    from config import config
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline(config)
    print(f"{_c('✓', GREEN)} Pipeline chargé.")
    return pipeline, config


def query_direct(pipeline, question: str, history: list = None) -> dict:
    result = pipeline.query(
        question=question,
        use_query_transform=False,
        stream=False,
        history=history or None,
    )
    return result


# ─── Mode API ────────────────────────────────────────────────────────────────

def get_token(api_url: str, email: str, password: str) -> str:
    import requests
    r = requests.post(f"{api_url}/auth/login", json={"email": email, "password": password})
    r.raise_for_status()
    return r.json()["token"]


def query_api(api_url: str, token: str, question: str) -> dict:
    import requests
    r = requests.post(
        f"{api_url}/query",
        json={"question": question, "history": []},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


# ─── Affichage ───────────────────────────────────────────────────────────────

def print_separator(char="─", width=70):
    print(char * width)


def print_result(test: dict, answer: str, scores: dict, verbose: bool = False):
    qid    = test.get("id", "?")
    cat    = test.get("category", "?")
    kw_s   = scores.get("keyword")
    cont_s = scores.get("containment")
    rl_s   = scores.get("rouge_l")
    sem_s  = scores.get("semantic")
    llm_s  = scores.get("llm_judge")
    overall = scores.get("overall", 0)

    icon = _c("✓", GREEN) if overall >= 0.7 else (_c("~", YELLOW) if overall >= 0.4 else _c("✗", RED))
    color = GREEN if overall >= 0.7 else (YELLOW if overall >= 0.4 else RED)

    print(f"\n{_c(f'[{qid}]', CYAN)} {_c(cat.upper(), BLUE)} — {test['question'][:80]}")
    print(f"  {icon} Score global : {_c(f'{overall:.0%}', color)}", end="")
    if kw_s   is not None: print(f"  | Mots-clés : {kw_s:.0%}", end="")
    if cont_s is not None: print(f"  | Containment : {cont_s:.0%}", end="")
    if rl_s   is not None: print(f"  | ROUGE-L : {rl_s:.0%}", end="")
    if sem_s  is not None: print(f"  | Sémantique : {sem_s:.2f}", end="")
    if llm_s  is not None: print(f"  | LLM-juge : {llm_s:.0%}", end="")
    print()

    if verbose:
        print(f"  {_c('Réponse :', BOLD)} {answer[:300]}{'...' if len(answer) > 300 else ''}")


def print_summary(results: list, elapsed: float, use_llm_judge: bool):
    print_separator("═")
    print(f"{_c(BOLD + 'RAPPORT DE PERFORMANCE', BOLD)}")
    print_separator("═")

    total = len(results)
    if total == 0:
        print("Aucun test exécuté.")
        return

    correct   = sum(1 for r in results if r["scores"]["overall"] >= 0.7)
    partial   = sum(1 for r in results if 0.4 <= r["scores"]["overall"] < 0.7)
    incorrect = sum(1 for r in results if r["scores"]["overall"] < 0.4)

    print(f"\n  Tests exécutés : {_c(total, BOLD)}")
    print(f"  {_c('✓ Corrects', GREEN)}  (≥ 70%) : {_c(correct, GREEN)}  ({correct/total:.0%})")
    print(f"  {_c('~ Partiels', YELLOW)} (40-70%) : {_c(partial, YELLOW)}  ({partial/total:.0%})")
    print(f"  {_c('✗ Incorrects', RED)} (< 40%)  : {_c(incorrect, RED)}  ({incorrect/total:.0%})")

    avg_overall = sum(r["scores"]["overall"] for r in results) / total
    print(f"\n  {_c('Score moyen global :', BOLD)} {_c(f'{avg_overall:.1%}', GREEN if avg_overall >= 0.7 else YELLOW)}")

    avg_kw = [r["scores"]["keyword"] for r in results if r["scores"]["keyword"] is not None]
    if avg_kw:
        print(f"  Score moyen mots-clés  : {sum(avg_kw)/len(avg_kw):.1%}")

    avg_rl = [r["scores"]["rouge_l"] for r in results if r["scores"]["rouge_l"] is not None]
    if avg_rl:
        print(f"  Score moyen ROUGE-L    : {sum(avg_rl)/len(avg_rl):.1%}")

    avg_sem = [r["scores"]["semantic"] for r in results if r["scores"]["semantic"] is not None]
    if avg_sem:
        print(f"  Similarité sémantique  : {sum(avg_sem)/len(avg_sem):.2f}")

    if use_llm_judge:
        avg_llm = [r["scores"]["llm_judge"] for r in results if r["scores"]["llm_judge"] is not None]
        if avg_llm:
            print(f"  Score LLM-juge moyen   : {sum(avg_llm)/len(avg_llm):.1%}")

    # Par catégorie
    categories = {}
    for r in results:
        cat = r["test"]["category"]
        categories.setdefault(cat, []).append(r["scores"]["overall"])

    if len(categories) > 1:
        print(f"\n  {_c('Par catégorie :', BOLD)}")
        for cat, scores in sorted(categories.items()):
            avg = sum(scores) / len(scores)
            color = GREEN if avg >= 0.7 else (YELLOW if avg >= 0.4 else RED)
            print(f"    {cat:<15} : {_c(f'{avg:.0%}', color)} ({len(scores)} tests)")

    print(f"\n  Durée totale : {elapsed:.1f}s  |  Durée moy/test : {elapsed/total:.1f}s")
    print_separator("═")


# ─── Calcul du score global ────────────────────────────────────────────────────

def compute_overall(scores: dict) -> float:
    """
    Calcule un score global calibré pour les réponses longues.

    Logique :
    - LLM-juge disponible → il domine (le plus fiable).
    - expected_contains fourni :
        * containment=1.0 → la phrase attendue EST dans la réponse → bonne réponse.
          Score = 0.4 + 0.6 * keyword (min 40%, modulé par la couverture des mots-clés).
        * containment=0.0 → phrase attendue ABSENTE → mauvaise réponse, max 50%.
    - Pas d'expected_contains → score = keyword seul (heuristique simple).

    NOTE : ROUGE-L et sémantique sont ignorés car `expected_contains` est généralement
    une courte phrase de référence, pas une réponse complète. Les comparer à une longue
    réponse donne ROUGE-L ≈ 0% et similarité faible même quand la réponse est correcte.
    """
    kw_s   = scores.get("keyword")
    cont_s = scores.get("containment")  # None si pas d'expected_contains
    sem_s  = scores.get("semantic")
    llm_s  = scores.get("llm_judge")

    # LLM juge : fiable, domine tout
    if llm_s is not None:
        kw = kw_s if kw_s is not None else 0.5
        return min(1.0, 0.7 * llm_s + 0.3 * kw)

    # expected_contains fourni
    if cont_s is not None:
        if cont_s == 1.0:
            # Phrase attendue présente → réponse correcte
            kw = kw_s if kw_s is not None else 1.0
            return min(1.0, 0.4 + 0.6 * kw)
        else:
            # Phrase attendue absente → crédit partiel via keywords + sémantique
            kw  = kw_s  if kw_s  is not None else 0.0
            sem = sem_s if sem_s is not None else 0.0
            return min(0.50, 0.30 * kw + 0.20 * sem)

    # Pas d'expected_contains → mots-clés seuls
    if kw_s is not None:
        return kw_s

    return 0.0


# ─── Entrée principale ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Évaluation de performance du chatbot Sonatrach")
    parser.add_argument("--dataset",   default="test_dataset.json", help="Fichier JSON du dataset de test")
    parser.add_argument("--output",    default=None,                help="Fichier de sortie JSON pour le rapport")
    parser.add_argument("--category",  default=None,                help="Filtrer par catégorie (structured|rag|mixed)")
    parser.add_argument("--api-url",   default=None,                help="URL du serveur (ex: http://localhost:8001)")
    parser.add_argument("--email",     default=None,                help="Email pour authentification API")
    parser.add_argument("--password",  default=None,                help="Mot de passe pour authentification API")
    parser.add_argument("--llm-judge", action="store_true",         help="Utiliser le LLM local comme juge")
    parser.add_argument("--semantic",  action="store_true",         help="Calculer la similarité sémantique")
    parser.add_argument("--verbose",   action="store_true",         help="Afficher les réponses complètes")
    args = parser.parse_args()

    # Chargement du dataset
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.path.dirname(__file__), dataset_path)

    if not os.path.exists(dataset_path):
        print(f"{_c('ERREUR', RED)}: Dataset introuvable : {dataset_path}")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    if args.category:
        tests = [t for t in tests if t.get("category") == args.category]

    if not tests:
        print(f"{_c('ERREUR', RED)}: Aucun test trouvé (catégorie={args.category}).")
        sys.exit(1)

    print_separator("═")
    print(f"{_c(BOLD + 'ÉVALUATION DU CHATBOT SONATRACH', BOLD)}")
    print_separator("═")
    print(f"  Dataset : {dataset_path}  ({len(tests)} tests)")
    print(f"  Mode    : {'API → ' + args.api_url if args.api_url else 'Direct (pipeline Python)'}")
    if args.category:
        print(f"  Filtre  : catégorie={args.category}")
    print_separator()

    # ── Setup ────────────────────────────────────────────────────────────────

    pipeline = token = None
    use_api = bool(args.api_url)

    if use_api:
        if not args.email or not args.password:
            print(f"{_c('ERREUR', RED)}: --email et --password requis en mode API.")
            sys.exit(1)
        print(f"{_c('→', BLUE)} Authentification sur {args.api_url}...")
        import requests
        try:
            token = get_token(args.api_url, args.email, args.password)
            print(f"{_c('✓', GREEN)} Token obtenu.")
        except Exception as e:
            print(f"{_c('ERREUR', RED)}: Authentification échouée : {e}")
            sys.exit(1)
    else:
        pipeline, config = load_pipeline()

    # Embedding model pour similarité sémantique
    embed_model = None
    if args.semantic or (not args.llm_judge and not use_api):
        try:
            print(f"{_c('→', BLUE)} Chargement du modèle d'embedding pour la similarité sémantique...")
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            print(f"{_c('✓', GREEN)} Modèle embedding chargé.")
        except Exception as e:
            print(f"{_c('⚠', YELLOW)} Embedding non disponible : {e}")

    # LLM juge
    llm_model = llm_tokenizer = None
    if args.llm_judge:
        try:
            print(f"{_c('→', BLUE)} Chargement du LLM juge...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            from config import config as cfg
            model_id = cfg.llm_model
            llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            llm_model = llm_model.to(device)
            llm_model.eval()
            print(f"{_c('✓', GREEN)} LLM juge chargé sur {device}.")
        except Exception as e:
            print(f"{_c('⚠', YELLOW)} LLM juge non disponible : {e}")
            llm_model = llm_tokenizer = None

    # ── Boucle d'évaluation ────────────────────────────────────────────────

    results = []
    start_total = time.time()

    for i, test in enumerate(tests, 1):
        qid      = test.get("id", f"T{i:03d}")
        question = test["question"]
        keywords = test.get("expected_keywords") or []
        expected = test.get("expected_contains")

        print(f"\n{_c(f'[{i}/{len(tests)}]', CYAN)} {_c(qid, BOLD)} — {question[:80]}")

        t0 = time.time()
        answer = ""
        error  = None

        try:
            if use_api:
                res    = query_api(args.api_url, token, question)
                answer = res.get("answer", "")
            else:
                res    = query_direct(pipeline, question)
                answer = res.get("answer", "")
        except Exception as e:
            error  = str(e)
            answer = ""
            print(f"  {_c('ERREUR', RED)}: {e}")

        elapsed_q = time.time() - t0

        # Métriques
        scores = {}
        scores["keyword"]     = keyword_score(answer, keywords) if keywords else None
        scores["containment"] = containment_score(answer, expected)
        scores["rouge_l"]     = rouge_l_score(answer, expected) if expected else None
        scores["semantic"]    = semantic_score(answer, expected, embed_model) if (embed_model and expected) else None
        scores["llm_judge"]   = llm_judge_score(question, answer, llm_model, llm_tokenizer) if (llm_model and answer) else None
        scores["overall"]     = compute_overall(scores) if not error else 0.0

        print_result(test, answer, scores, verbose=args.verbose)
        print(f"  {_c('Temps :', BLUE)} {elapsed_q:.1f}s")

        results.append({
            "test":    test,
            "answer":  answer,
            "scores":  scores,
            "elapsed": elapsed_q,
            "error":   error,
        })

    elapsed_total = time.time() - start_total
    print_summary(results, elapsed_total, use_llm_judge=bool(llm_model))

    # ── Sauvegarde rapport ────────────────────────────────────────────────

    if args.output:
        report = {
            "dataset":       args.dataset,
            "total_tests":   len(results),
            "elapsed":       elapsed_total,
            "avg_overall":   sum(r["scores"]["overall"] for r in results) / len(results),
            "accuracy_70":   sum(1 for r in results if r["scores"]["overall"] >= 0.7) / len(results),
            "results":       results,
        }
        out_path = args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n{_c('→ Rapport sauvegardé :', BLUE)} {out_path}")


if __name__ == "__main__":
    main()
