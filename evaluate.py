"""
[ÉTAPE 2] Évaluation du chatbot.

Pour chaque question de test_questions.json, on mesure :
  - RETRIEVAL : le contexte retrouvé contient-il au moins un mot-clé attendu ?
  - RÉPONSE   : la réponse du modèle contient-elle au moins un mot-clé attendu ?

Usage :
    python evaluate.py              # retrieval + réponses (lent : le modèle génère)
    python evaluate.py --retrieval  # retrieval seul (rapide, sans le modèle de langage)
    python evaluate.py --ville Ottawa            # une seule ville
    python evaluate.py --retrieval --ville "Le Caire"
"""
import csv
import json
import os
import re
import sys
import time
import unicodedata

# Importer app.py charge les modèles mais NE lance PAS l'interface Gradio.
from app import retrieve_context, repondre, DATA_FOLDER

QUESTIONS_FILE = "test_questions.json"
RESULTS_FILE = "evaluation_resultats.csv"


def normaliser(texte: str) -> str:
    """Minuscules, sans accents, espaces ni ponctuation ('4 026 000' = '4026000', 'Al-Hilal' = 'Al Hilal')."""
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", texte.lower())


def contient_un_mot_cle(texte: str, mots_cles) -> bool:
    t = normaliser(texte)
    return any(normaliser(m) in t for m in mots_cles)


def mot_cle_dans_fichier(ville: str, mots_cles) -> bool:
    """Vérifie que la question de test est valide : la réponse existe bien dans le fichier de la ville."""
    chemin = os.path.join(DATA_FOLDER, f"{ville}_nettoye.txt")
    if not os.path.exists(chemin):
        return False
    with open(chemin, encoding="utf-8") as f:
        return contient_un_mot_cle(f.read(), mots_cles)


def main():
    retrieval_seul = "--retrieval" in sys.argv

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    # [ÉTAPE 3b] filtrer sur une ville : --ville Ottawa
    if "--ville" in sys.argv:
        ville_choisie = sys.argv[sys.argv.index("--ville") + 1]
        questions = [q for q in questions if q["ville"].lower() == ville_choisie.lower()]
        print(f"Ville : {ville_choisie} ({len(questions)} questions)")

    lignes = []
    ok_retrieval = ok_reponse = ok_les_deux = valides = 0
    durees = []

    for i, q in enumerate(questions, 1):
        ville, question, mots = q["ville"], q["question"], q["mots_cles"]

        if not mot_cle_dans_fichier(ville, mots):
            print(f"[{i}/{len(questions)}] ⚠ IGNORÉE (mot-clé absent du fichier {ville}) : {question}")
            continue
        valides += 1

        debut = time.time()
        if retrieval_seul:
            contexte = retrieve_context(question, ville)
            reponse = ""
        else:
            reponse, contexte = repondre(question, ville)
        duree = time.time() - debut

        r_ok = contient_un_mot_cle(contexte, mots)
        a_ok = None if retrieval_seul else contient_un_mot_cle(reponse, mots)
        ok_retrieval += r_ok
        ok_reponse += bool(a_ok)
        ok_les_deux += bool(r_ok and a_ok)   # [ÉTAPE 6]
        durees.append(duree)

        etat_r = "✅" if r_ok else "❌"
        etat_a = "" if retrieval_seul else (" | Réponse " + ("✅" if a_ok else "❌"))
        print(f"[{i}/{len(questions)}] {ville} | Retrieval {etat_r}{etat_a} | {duree:.1f}s | {question}")

        lignes.append({
            "ville": ville, "question": question, "mots_cles": " / ".join(mots),
            "retrieval_ok": r_ok, "reponse_ok": a_ok, "duree_s": round(duree, 1),
            "reponse": reponse.replace("\n", " "),
        })

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8-sig") as f:
        if lignes:
            w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(lignes)

    print("\n" + "=" * 50)
    print(f"Questions valides : {valides}/{len(questions)}")
    if valides:
        print(f"Retrieval : {ok_retrieval}/{valides} ({100 * ok_retrieval / valides:.0f} %)")
        if not retrieval_seul:
            print(f"Réponses  : {ok_reponse}/{valides} ({100 * ok_reponse / valides:.0f} %)")
            # [ÉTAPE 6] qualité du modèle de langage seul : quand le bon passage lui est fourni, répond-il juste ?
            if ok_retrieval:
                print(f"Réponses justes quand le bon passage est fourni : {ok_les_deux}/{ok_retrieval} ({100 * ok_les_deux / ok_retrieval:.0f} %)")
            print(f"Durée moyenne par réponse : {sum(durees) / len(durees):.0f} s")
    print(f"Détails dans : {RESULTS_FILE}")


if __name__ == "__main__":
    main()