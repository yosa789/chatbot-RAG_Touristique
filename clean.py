import os
import re
from pypdf import PdfReader
 
# Lignes parasites typiques des PDF Wikipédia (cartes, graphiques, légendes)
LIGNES_PARASITES = [
    r"Géolocalisation sur la carte\s*:.*",
    r"Histogramme.*",
    r"Diagramme climatique",
    r"Moyennes\s*:.*",
    r"Sources?\s*:?",
    r"Ce document provient de.*",
]


def couper_references(text: str) -> str:
    """[ÉTAPE 1] Supprime tout ce qui suit le début de la liste de références."""
    # La liste de références commence par une ligne "1. « Titre..." ou "1. Auteur..."
    m = re.search(r"(?m)^\s*1\.\s+\S", text)
    if m and m.start() > len(text) * 0.3:  # sécurité : pas au tout début du texte
        return text[: m.start()]
    # Sinon, on coupe sur le titre de section
    m = re.search(r"(?m)^\s*(Notes et références|Références)\s*$", text)
    if m:
        return text[: m.start()]
    return text


def clean_text_for_rag(text: str) -> str:
    """Nettoie le texte extrait d'un PDF Wikipédia pour l'injection dans un pipeline RAG."""
    if not isinstance(text, str):
        return ""

    # 1. Couper les références et tout ce qui suit
    text = couper_references(text)

    # 2. Supprimer les marqueurs de page ajoutés à l'extraction
    text = re.sub(r"(?m)^\s*--- Page \d+ ---\s*$", "", text)

    # 3. Supprimer les balises de source / citation / "réf. nécessaire"
    text = re.sub(r"\[(?:source|cite):\s*\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\[\s*réf\.\s*nécessaire\s*\]", "", text)

    # 4. Supprimer les URL
    text = re.sub(r"http[s]?://\S+", "", text)

    # 5. Supprimer les lignes qui ne contiennent QUE des numéros de notes
    #    ex: "13 14", "6, 7", "1"  (les nombres comme "19 000" sont conservés)
    text = re.sub(r"(?m)^\s*\d{1,2}(?:\s*,\s*\d{1,2}|\s+\d{1,2})*\s*$\n?", "", text)

    # 6. Supprimer les lignes parasites (cartes, légendes de graphiques...)
    for motif in LIGNES_PARASITES:
        text = re.sub(rf"(?m)^\s*{motif}\s*$\n?", "", text)

    # 7. Corriger les césures de mots en fin de ligne
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 8. Harmoniser les espaces et sauts de ligne
    text = re.sub(r"\r?\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # 9. Espaces avant la ponctuation
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)

    return text.strip()


def process_pdf_folder(data_folder: str):
    """Parcourt le dossier data, lit chaque PDF, extrait et nettoie le texte."""
    if not os.path.exists(data_folder):
        print(f"Erreur : Le dossier '{data_folder}' n'existe pas.")
        return

    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
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

                output_filename = filename.replace(".pdf", "_nettoye.txt")
                output_path = os.path.join(data_folder, output_filename)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                print(f"-> Fichier sauvegardé : {output_filename}")

            except Exception as e:
                print(f"Erreur lors de la lecture de {filename}: {e}")


if __name__ == "__main__":
    process_pdf_folder("./data")