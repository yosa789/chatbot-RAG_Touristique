import os
import re
from pypdf import PdfReader

from config import DATA_FOLDER

LIGNES_PARASITES = [
    r"Géolocalisation sur la carte\s*:.*",
    r"Histogramme.*",
    r"Diagramme climatique",
    r"Moyennes\s*:.*",
    r"Sources?\s*:?",
    r"Ce document provient de.*",
]

MAJUSCULE = r"[A-ZÀ-ÖØ-Þ]"


def couper_references(text: str) -> str:
    """Supprime la liste de références et tout ce qui suit."""
    # 1. Titre de section (fiable)
    m = re.search(r"(?m)^\s*(Notes et références|Références)\s*$", text)
    if m:
        return text[: m.start()]
    for m in re.finditer(r"(?m)^\s*1\.\s+\S", text):
        if m.start() > len(text) * 0.5:
            return text[: m.start()]
    return text


def _cesure(m: re.Match) -> str:
    """'ville-\\nneuve' -> 'villeneuve' ; mais 'Al-\\nAzhar' -> 'Al-Azhar' (tiret conservé)."""
    debut, fin = m.group(1), m.group(2)
    return debut + fin if fin[0].islower() else f"{debut}-{fin}"


def recoller_paragraphes(text: str) -> str:
    """Recolle les lignes d'un même paragraphe ; garde une ligne vide (\\n\\n) entre paragraphes.
    Le splitter peut ainsi couper entre paragraphes plutôt qu'au milieu d'une phrase."""
    lignes = [l.strip() for l in text.split("\n") if l.strip()]
    paragraphes, courant = [], ""
    for ligne in lignes:
        fin_phrase = re.search(r"[.!?»)]$", courant)
        titre = len(courant) < 50 and not re.search(r"[.!?»),;:]$", courant)
        if courant and (fin_phrase or titre) and re.match(MAJUSCULE, ligne):
            paragraphes.append(courant)
            courant = ligne
        else:
            courant = f"{courant} {ligne}" if courant else ligne
    if courant:
        paragraphes.append(courant)
    return "\n\n".join(paragraphes)


def clean_text_for_rag(text: str) -> str:
    """Nettoie le texte extrait d'un PDF Wikipédia pour l'injection dans un pipeline RAG."""
    if not isinstance(text, str):
        return ""

    text = couper_references(text)

    # Marqueurs de page
    text = re.sub(r"(?m)^\s*--- Page \d+ ---\s*$", "", text)

    # Balises de source / citation / "réf. nécessaire"
    text = re.sub(r"\[(?:source|cite):\s*\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\[\s*réf\.\s*nécessaire\s*\]", "", text)

    # URL
    text = re.sub(r"http[s]?://\S+", "", text)

    # Lignes qui ne contiennent QUE des numéros de notes ("13 14", "6, 7", "1")
    text = re.sub(r"(?m)^\s*\d{1,2}(?:\s*,\s*\d{1,2}|\s+\d{1,2})*\s*$\n?", "", text)

    # Lignes parasites (cartes, légendes de graphiques...)
    for motif in LIGNES_PARASITES:
        text = re.sub(rf"(?m)^\s*{motif}\s*$\n?", "", text)

    # Césures en fin de ligne (avant de recoller les lignes)
    text = re.sub(r"(\w+)-[ \t]*\n[ \t]*(\w+)", _cesure, text)

    # Paragraphes
    text = re.sub(r"\r", "", text)
    text = recoller_paragraphes(text)

    # Espaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+([?.!,;:])", r"\1", text)

    return text.strip()


def process_pdf_folder(data_folder: str):
    """Parcourt le dossier data, lit chaque PDF, extrait et nettoie le texte."""
    if not os.path.exists(data_folder):
        print(f"Erreur : Le dossier '{data_folder}' n'existe pas.")
        return

    for filename in os.listdir(data_folder):
        if not filename.endswith(".pdf"):
            continue
        pdf_path = os.path.join(data_folder, filename)
        print(f"\nTraitement du fichier : {filename}...")
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += "\n" + text

            cleaned_text = clean_text_for_rag(full_text)
            print(f"-> Succès ! {len(cleaned_text)} caractères nettoyés.")
            print("Aperçu :")
            print(cleaned_text[:300] + "...\n")

            output_filename = filename[: -len(".pdf")] + "_nettoye.txt"
            with open(os.path.join(data_folder, output_filename), "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"-> Fichier sauvegardé : {output_filename}")
        except Exception as e:
            print(f"Erreur lors de la lecture de {filename}: {e}")


if __name__ == "__main__":
    process_pdf_folder(DATA_FOLDER)