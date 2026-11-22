"""
Évaluation du chatbot.

Pour chaque question de test_questions.json, on mesure :
  - ROUTAGE   : detecter_ville() retrouve-t-elle la bonne ville depuis la question seule ?
  - RETRIEVAL : le contexte retrouvé contient-il au moins un mot-clé attendu ?
  - RÉPONSE   : la réponse du modèle contient-elle au moins un mot-clé attendu ?

Usage :
    python evaluate.py                          # tout (lent : le modèle génère)
    python evaluate.py --retrieval              # sans charger le modèle de langage
    python evaluate.py --ville Ottawa
    python evaluate.py --retrieval --ville "Le Caire"
"""
import argparse
import csv
import json
import os
import re
import time
import unicodedata

from app import retrieve_context, repondre, detecter_ville, TOUTES
from config import DATA_FOLDER

QUESTIONS_FILE = "test_questions.json"
RESULTS_FILE = "evaluation_resultats.csv"


def normaliser(texte: str) -> str:
    """Minuscules, sans accents ni ponctuation ('4 026 000' = '4026000', 'Al-Hilal' = 'Al Hilal')."""
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", texte.lower())


def contient_un_mot_cle(texte: str, mots_cles) -> bool:
    t = normaliser(texte)
    for m in mots_cles:
        mot = normaliser(m)
        if mot.isdigit():
            if re.search(rf"(?<!\d){mot}(?!\d)", t):
                return True
        elif mot in t:
            return True
    return False


def mot_cle_dans_fichier(ville: str, mots_cles) -> bool:
    """Vérifie que la question est valide : la réponse existe bien dans le fichier de la ville."""
    chemin = os.path.join(DATA_FOLDER, f"{ville}_nettoye.txt")
    if not os.path.exists(chemin):
        return False
    with open(chemin, encoding="utf-8") as f:
        return contient_un_mot_cle(f.read(), mots_cles)


def main():
    parser = argparse.ArgumentParser(description="Évaluation du chatbot")
    parser.add_argument("--retrieval", action="store_true", help="retrieval seul, sans modèle de langage")
    parser.add_argument("--ville", help="évaluer une seule ville")
    args = parser.parse_args()

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    if args.ville:
        questions = [q for q in questions if q["ville"].lower() == args.ville.lower()]
        print(f"Ville : {args.ville} ({len(questions)} questions)")

    lignes = []
    ok_retrieval = ok_reponse = ok_les_deux = valides = 0
    ignorees = []
    routage_ok = routage_vide = 0
    durees = []

    for i, q in enumerate(questions, 1):
        ville, question, mots = q["ville"], q["question"], q["mots_cles"]

        if not mot_cle_dans_fichier(ville, mots):
            ignorees.append(question)
            print(f"[{i}/{len(questions)}] ⚠ IGNORÉE (mot-clé absent du fichier {ville}) : {question}")
            continue
        valides += 1

        # Routage : la ville est-elle devinée à partir de la question seule ?
        ville_detectee = detecter_ville(question)
        routage_ok += ville_detectee == ville
        routage_vide += ville_detectee == TOUTES

        debut = time.time()
        if args.retrieval:
            contexte, reponse = retrieve_context(question, ville), ""
        else:
            reponse, contexte = repondre(question, ville)
        duree = time.time() - debut

        r_ok = contient_un_mot_cle(contexte, mots)
        a_ok = None if args.retrieval else contient_un_mot_cle(reponse, mots)
        ok_retrieval += r_ok
        ok_reponse += bool(a_ok)
        ok_les_deux += bool(r_ok and a_ok)
        durees.append(duree)

        etat_a = "" if args.retrieval else " | Réponse " + ("✅" if a_ok else "❌")
        print(f"[{i}/{len(questions)}] {ville} | Retrieval {'✅' if r_ok else '❌'}{etat_a} | {duree:.1f}s | {question}")

        lignes.append({
            "ville": ville, "question": question, "mots_cles": " / ".join(mots),
            "ville_detectee": ville_detectee,
            "retrieval_ok": r_ok, "reponse_ok": a_ok, "duree_s": round(duree, 1),
            "reponse": reponse.replace("\n", " "),
        })

    if lignes:
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(lignes)

    print("\n" + "=" * 50)
    print(f"Questions écrites : {len(questions)} | valides : {valides} | ignorées : {len(ignorees)}")
    if ignorees:
        print("  -> à corriger dans test_questions.json (mot-clé introuvable dans le fichier)")
    if valides:
        print(f"Routage   : {routage_ok}/{valides} bien détectées, {routage_vide} sans ville détectée "
              f"(→ « {TOUTES} »), {valides - routage_ok - routage_vide} mauvaise ville")
        print(f"Retrieval : {ok_retrieval}/{valides} ({100 * ok_retrieval / valides:.0f} %)")
        if not args.retrieval:
            print(f"Réponses  : {ok_reponse}/{valides} ({100 * ok_reponse / valides:.0f} %)")
            if ok_retrieval:
                print(f"Réponses justes quand le bon passage est fourni : "
                      f"{ok_les_deux}/{ok_retrieval} ({100 * ok_les_deux / ok_retrieval:.0f} %)")
            print(f"Durée moyenne par réponse : {sum(durees) / len(durees):.1f} s")
    print(f"Détails dans : {RESULTS_FILE}")


if __name__ == "__main__":
    main()