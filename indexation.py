import os
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma            # pip install langchain-chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import DATA_FOLDER, PERSIST_DIRECTORY, EMBEDDING_MODEL

CHUNK_SIZE = 500     # le modèle multilingue lit ~128 tokens max
CHUNK_OVERLAP = 100


def build_vector_database():
    # 1. Charger les fichiers _nettoye.txt et noter la ville de chacun
    documents = []
    for filename in sorted(os.listdir(DATA_FOLDER)):
        if filename.endswith("_nettoye.txt"):
            ville = filename[: -len("_nettoye.txt")]
            docs_ville = TextLoader(os.path.join(DATA_FOLDER, filename), encoding="utf-8").load()
            for d in docs_ville:
                d.metadata["ville"] = ville
            documents.extend(docs_ville)

    if not documents:
        print("Aucun fichier _nettoye.txt trouvé dans le dossier data.")
        return
    print(f"{len(documents)} documents chargés pour l'indexation.")

    # 2. Découpage
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents(documents)

    # Chaque morceau commence par le nom de la ville : il "sait" de quelle ville il parle
    for d in docs:
        d.page_content = f"Ville : {d.metadata['ville']}\n{d.page_content}"
    print(f"Texte découpé en {len(docs)} morceaux (chunks).")

    # 3. Repartir d'une base vide : sinon un second lancement duplique tous les chunks
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

    print("Chargement du modèle d'embedding Hugging Face...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Création de la base vectorielle ChromaDB...")
    Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
    print(f"-> Base vectorielle créée dans : {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    build_vector_database()