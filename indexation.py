import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def build_vector_database():
    data_folder = "./data"
    persist_directory = "./chroma_db"

    # 1. Charger les fichiers _nettoye.txt et noter la ville de chacun
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith("_nettoye.txt"):
            ville = filename[: -len("_nettoye.txt")]
            file_path = os.path.join(data_folder, filename)
            docs_ville = TextLoader(file_path, encoding="utf-8").load()
            for d in docs_ville:
                d.metadata["ville"] = ville  # [ÉTAPE 1] métadonnée ville
            documents.extend(docs_ville)

    if not documents:
        print("Aucun fichier _nettoye.txt trouvé dans le dossier data.")
        return

    print(f"{len(documents)} documents chargés pour l'indexation.")

    # 2. Découpage en morceaux plus petits et plus précis  [ÉTAPE 1]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # [ÉTAPE 3] avant : 800 (le modèle multilingue lit ~128 tokens max)
        chunk_overlap=100,  # [ÉTAPE 3] avant : 150
    )
    docs = text_splitter.split_documents(documents)

    # [ÉTAPE 1] Chaque morceau commence par le nom de la ville : il "sait" de quelle ville il parle
    for d in docs:
        d.page_content = f"Ville : {d.metadata['ville']}\n{d.page_content}"

    print(f"Texte découpé en {len(docs)} morceaux (chunks).")

    # 3. Modèle d'embedding (doit rester identique à celui de app.py)
    print("Chargement du modèle d'embedding Hugging Face...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # [ÉTAPE 3] multilingue
    )

    # 4. Base ChromaDB
    print("Création de la base vectorielle ChromaDB...")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"-> Base vectorielle créée dans : {persist_directory}")


if __name__ == "__main__":
    build_vector_database()