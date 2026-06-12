#!/usr/bin/env python3
"""
quick_eval.py — Évaluation SANS dataset de référence.

Fonctionne avec une simple liste de questions (fichier texte).
Évalue automatiquement chaque réponse sur 4 critères sans avoir besoin
de connaître les réponses attendues — idéal pour les réponses longues PDF.

Usage :
  python quick_eval.py --questions questions.txt
  python quick_eval.py --questions questions.txt --llm-judge
  python quick_eval.py --questions questions.txt --output rapport.json
  python quick_eval.py --questions questions.txt --api-url http://localhost:8001 --email x --password y
"""

import sys, os, json, time, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import types
if "langchain" not in sys.modules:
    _lc_stub = types.ModuleType("langchain")
    _lc_stub.debug = False
    sys.modules["langchain"] = _lc_stub

# ─── ANSI couleurs ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; RED   = "\033[91m"; YELLOW = "\033[93m"
BLUE   = "\033[94m"; CYAN  = "\033[96m"; BOLD   = "\033[1m"; RESET = "\033[0m"
def _c(t, col): return f"{col}{t}{RESET}"


# ─── Métriques sans référence ─────────────────────────────────────────────────

# Phrases indiquant que le bot ne sait pas répondre
_REFUSAL_PATTERNS = [
    r"je\s+n[e']?\s*(ai|sais|trouve|dispose|peux|parviens)\s+pas",
    r"aucune\s+(information|donnée|réponse|résultat)",
    r"je\s+ne\s+suis\s+pas\s+en\s+mesure",
    r"désolé",
    r"pas\s+de\s+(résultat|donnée|information)",
    r"n[o']?\s+information",
    r"malheureusement",
    r"hors\s+de\s+ma\s+connaissance",
    r"non\s+disponible",
]

def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(re.search(p, low) for p in _REFUSAL_PATTERNS)


def length_score(answer: str) -> tuple[float, str]:
    """Score basé sur la longueur de la réponse."""
    words = len(answer.split())
    if words < 10:
        return 0.0, f"trop court ({words} mots)"
    elif words < 30:
        return 0.4, f"court ({words} mots)"
    elif words < 80:
        return 0.7, f"moyen ({words} mots)"
    else:
        return 1.0, f"complet ({words} mots)"


def relevance_score(question: str, answer: str) -> float:
    """
    Chevauchement de tokens entre la question et la réponse.
    Approximation simple de la pertinence sans LLM.
    """
    import unicodedata

    def tokens(text):
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        stopwords = {"le", "la", "les", "de", "du", "des", "un", "une", "est",
                     "et", "en", "à", "au", "aux", "pour", "par", "qui", "que",
                     "quel", "quelle", "quels", "quelles", "comment", "combien",
                     "quoi", "se", "son", "sa", "ses", "ce", "cet", "cette", "ces",
                     "il", "elle", "ils", "elles", "je", "vous", "nous", "on"}
        return {w for w in re.findall(r"\w+", text) if len(w) > 2 and w not in stopwords}

    q_tokens = tokens(question)
    a_tokens = tokens(answer)
    if not q_tokens:
        return 0.5
    overlap = len(q_tokens & a_tokens)
    return min(1.0, overlap / len(q_tokens))


def structure_score(answer: str) -> float:
    """
    Score de structure : réponses bien structurées (listes, étapes, chiffres)
    indiquent généralement une réponse de qualité.
    """
    score = 0.5
    # Présence de listes ou numérotation
    if re.search(r"(\d+[\.\)]\s|\•|\-\s|étape)", answer, re.I):
        score += 0.2
    # Présence de données concrètes (chiffres, dates, noms propres)
    if re.search(r"\d+", answer):
        score += 0.1
    # Longueur raisonnable de phrases
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 10]
    if len(sentences) >= 2:
        score += 0.2
    return min(1.0, score)


def has_source(result: dict) -> bool:
    """Vérifie si la réponse est accompagnée de sources documentaires."""
    sources = result.get("sources") or result.get("chunks_used") or []
    return len(sources) > 0


def llm_judge(question: str, answer: str, llm, tokenizer) -> tuple[float, str]:
    """
    Fait évaluer la réponse par le LLM sur une échelle 1-5 sur 3 critères.
    Retourne un score normalisé 0-1 et un commentaire.
    """
    prompt = f"""Tu es un expert en évaluation de chatbots d'entreprise. Évalue cette réponse sur 3 critères.

QUESTION : {question}

RÉPONSE DU CHATBOT :
{answer[:800]}

Évalue sur une échelle de 1 à 5 :
- PERTINENCE (la réponse répond-elle à la question ?) : X/5
- COMPLÉTUDE (la réponse est-elle suffisamment détaillée ?) : X/5
- COHÉRENCE (la réponse est-elle logique et bien rédigée ?) : X/5

Réponds UNIQUEMENT avec ce format exact :
PERTINENCE: X
COMPLÉTUDE: X
COHÉRENCE: X"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1200)
        if hasattr(llm, "device"):
            inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        out = llm.generate(
            **inputs, max_new_tokens=60, temperature=0.0,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        scores = {}
        for line in decoded.strip().split("\n"):
            m = re.match(r"(PERTINENCE|COMPLÉTUDE|COHÉRENCE)\s*:\s*(\d)", line.upper())
            if m:
                scores[m.group(1)] = int(m.group(2))

        if len(scores) >= 2:
            avg = sum(scores.values()) / len(scores)
            norm = (avg - 1) / 4  # normalise 1-5 → 0-1
            comment = " | ".join(f"{k}: {v}/5" for k, v in scores.items())
            return norm, comment
        return None, "Format non reconnu"
    except Exception as e:
        return None, f"Erreur : {e}"


# ─── Score global ─────────────────────────────────────────────────────────────

def compute_score(length_s, relevance_s, structure_s, llm_s=None, refusal=False) -> float:
    if refusal:
        return 0.05
    if llm_s is not None:
        # LLM disponible : il domine
        return 0.50 * llm_s + 0.20 * relevance_s + 0.15 * length_s + 0.15 * structure_s
    else:
        # Sans LLM : heuristiques seules
        return 0.40 * relevance_s + 0.35 * length_s + 0.25 * structure_s


# ─── Chargement des questions ─────────────────────────────────────────────────

def load_questions(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    questions = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            questions.append(line)
    return questions


# ─── Affichage ───────────────────────────────────────────────────────────────

def bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    color = GREEN if score >= 0.7 else (YELLOW if score >= 0.4 else RED)
    return _c("█" * filled, color) + "░" * (width - filled) + f" {score:.0%}"


def print_result(i, total, question, answer, scores_detail, overall, elapsed, verbose):
    icon = _c("✓", GREEN) if overall >= 0.7 else (_c("~", YELLOW) if overall >= 0.4 else _c("✗", RED))
    print(f"\n{_c(f'[{i}/{total}]', CYAN)} {question[:80]}")
    print(f"  {icon} {bar(overall)}  {_c(f'{elapsed:.1f}s', BLUE)}")
    for label, val in scores_detail.items():
        if val is not None:
            print(f"     {label:<22} {val}")
    if verbose and answer:
        print(f"\n  {_c('Réponse :', BOLD)}")
        for line in answer[:600].split("\n"):
            print(f"    {line}")
        if len(answer) > 600:
            print(f"    {_c('... (tronqué)', YELLOW)}")


def print_summary(results, elapsed_total, use_llm):
    n = len(results)
    if n == 0:
        return
    correct  = sum(1 for r in results if r["overall"] >= 0.7)
    partial  = sum(1 for r in results if 0.4 <= r["overall"] < 0.7)
    bad      = sum(1 for r in results if r["overall"] < 0.4)
    refusals = sum(1 for r in results if r.get("refusal"))
    avg      = sum(r["overall"] for r in results) / n

    print("\n" + "═" * 65)
    print(_c(BOLD + " RAPPORT D'ÉVALUATION — SANS RÉFÉRENCE", BOLD))
    print("═" * 65)
    print(f"\n  Tests : {_c(n, BOLD)}")
    print(f"  {_c('✓ Bonnes réponses', GREEN)}    (≥ 70%) : {_c(correct, GREEN)} ({correct/n:.0%})")
    print(f"  {_c('~ Réponses partielles', YELLOW)} (40-70%) : {_c(partial, YELLOW)} ({partial/n:.0%})")
    print(f"  {_c('✗ Mauvaises réponses', RED)}   (< 40%)  : {_c(bad, RED)} ({bad/n:.0%})")
    if refusals:
        print(f"  {_c('⊘ Refus / Pas de réponse', YELLOW)}  : {_c(refusals, YELLOW)}")

    print(f"\n  {_c('Score moyen :', BOLD)} {bar(avg, 25)}")

    avg_rel = [r["relevance"] for r in results]
    avg_len = [r["length_s"] for r in results]
    avg_str = [r["structure"] for r in results]
    print(f"\n  Pertinence moy.  : {sum(avg_rel)/n:.1%}")
    print(f"  Longueur moy.    : {sum(avg_len)/n:.1%}")
    print(f"  Structure moy.   : {sum(avg_str)/n:.1%}")
    if use_llm:
        llm_scores = [r["llm_score"] for r in results if r.get("llm_score") is not None]
        if llm_scores:
            print(f"  LLM-juge moy.    : {sum(llm_scores)/len(llm_scores):.1%}")

    sources_count = sum(1 for r in results if r.get("has_source"))
    print(f"  Avec sources     : {sources_count}/{n} ({sources_count/n:.0%})")
    print(f"\n  Durée totale : {elapsed_total:.1f}s  |  Moy/question : {elapsed_total/n:.1f}s")
    print("═" * 65)

    print(f"\n{_c('INTERPRÉTATION :', BOLD)}")
    if avg >= 0.7:
        print(f"  {_c('Excellente performance', GREEN)} — le chatbot répond bien à la plupart des questions.")
    elif avg >= 0.5:
        print(f"  {_c('Performance correcte', YELLOW)} — quelques questions reçoivent des réponses incomplètes.")
    else:
        print(f"  {_c('Performance insuffisante', RED)} — réviser les documents ou le pipeline RAG.")
    if refusals > n * 0.2:
        print(f"  {_c('⚠ Trop de refus', YELLOW)} ({refusals}) — vérifier que les documents sont bien ingérés.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="questions.txt")
    parser.add_argument("--output",    default=None)
    parser.add_argument("--api-url",   default=None)
    parser.add_argument("--email",     default=None)
    parser.add_argument("--password",  default=None)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    # Chargement questions
    qpath = args.questions
    if not os.path.isabs(qpath):
        qpath = os.path.join(os.path.dirname(__file__), qpath)
    if not os.path.exists(qpath):
        print(f"{_c('ERREUR', RED)}: Fichier introuvable : {qpath}")
        sys.exit(1)

    questions = load_questions(qpath)
    if not questions:
        print(f"{_c('ERREUR', RED)}: Aucune question trouvée dans {qpath}")
        sys.exit(1)

    print("═" * 65)
    print(_c(BOLD + " ÉVALUATION SANS DATASET — CHATBOT SONATRACH", BOLD))
    print("═" * 65)
    print(f"  Questions : {len(questions)}  |  Fichier : {qpath}")
    print(f"  Mode      : {'API → ' + args.api_url if args.api_url else 'Direct (pipeline Python)'}")
    print(f"  LLM-juge  : {'Oui' if args.llm_judge else 'Non (heuristiques)'}")
    print("─" * 65)

    # Setup pipeline ou API
    pipeline = token = None
    if args.api_url:
        if not args.email or not args.password:
            print(f"{_c('ERREUR', RED)}: --email et --password requis en mode API.")
            sys.exit(1)
        import requests
        try:
            r = requests.post(f"{args.api_url}/auth/login",
                              json={"email": args.email, "password": args.password})
            r.raise_for_status()
            token = r.json()["token"]
            print(f"{_c('✓', GREEN)} Authentifié sur {args.api_url}")
        except Exception as e:
            print(f"{_c('ERREUR', RED)}: {e}"); sys.exit(1)
    else:
        print(f"{_c('→', BLUE)} Chargement du pipeline RAG...")
        from config import config
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline(config)
        print(f"{_c('✓', GREEN)} Pipeline chargé.")

    # Setup LLM juge
    llm_model = llm_tokenizer = None
    if args.llm_judge:
        try:
            print(f"{_c('→', BLUE)} Chargement du LLM juge...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            from config import config as cfg
            llm_tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model)
            llm_model = AutoModelForCausalLM.from_pretrained(
                cfg.llm_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            llm_model.eval()
            print(f"{_c('✓', GREEN)} LLM juge prêt.")
        except Exception as e:
            print(f"{_c('⚠', YELLOW)} LLM juge non disponible : {e}")

    # ── Boucle ────────────────────────────────────────────────────────────────
    results = []
    t_total = time.time()

    for i, question in enumerate(questions, 1):
        t0 = time.time()
        answer = ""
        raw_result = {}
        error = None

        try:
            if args.api_url:
                import requests
                r = requests.post(
                    f"{args.api_url}/query",
                    json={"question": question, "history": []},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=120,
                )
                r.raise_for_status()
                raw_result = r.json()
            else:
                raw_result = pipeline.query(question=question, stream=False, history=None)
            answer = raw_result.get("answer", "")
        except Exception as e:
            error = str(e)
            print(f"  {_c('ERREUR', RED)}: {e}")

        elapsed = time.time() - t0

        # ── Calcul des métriques ─────────────────────────────────────────────
        refusal    = is_refusal(answer) if answer else True
        len_s, len_detail = length_score(answer)
        rel_s      = relevance_score(question, answer)
        str_s      = structure_score(answer)
        src        = has_source(raw_result)
        llm_s      = llm_comment = None

        if llm_model and answer and not refusal:
            llm_s, llm_comment = llm_judge(question, answer, llm_model, llm_tokenizer)

        overall = compute_score(len_s, rel_s, str_s, llm_s, refusal)

        scores_detail = {
            "Pertinence (tokens)  :": f"{rel_s:.0%}",
            "Longueur réponse     :": f"{len_s:.0%}  ({len_detail})",
            "Structure            :": f"{str_s:.0%}",
        }
        if src:
            scores_detail["Sources citées       :"] = _c("Oui", GREEN)
        else:
            scores_detail["Sources citées       :"] = _c("Non", YELLOW)
        if refusal:
            scores_detail["Statut               :"] = _c("⊘ REFUS / PAS DE RÉPONSE", RED)
        if llm_s is not None:
            scores_detail["LLM-juge             :"] = f"{llm_s:.0%}  ({llm_comment})"

        print_result(i, len(questions), question, answer, scores_detail, overall, elapsed, args.verbose)

        results.append({
            "question":  question,
            "answer":    answer,
            "overall":   overall,
            "relevance": rel_s,
            "length_s":  len_s,
            "structure": str_s,
            "llm_score": llm_s,
            "llm_comment": llm_comment,
            "has_source": src,
            "refusal":   refusal,
            "elapsed":   elapsed,
            "error":     error,
        })

    elapsed_total = time.time() - t_total
    print_summary(results, elapsed_total, use_llm=bool(llm_model))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({
                "questions_file": args.questions,
                "total":          len(results),
                "avg_overall":    sum(r["overall"] for r in results) / len(results),
                "results":        results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n{_c('→ Rapport sauvegardé :', BLUE)} {args.output}")


if __name__ == "__main__":
    main()
