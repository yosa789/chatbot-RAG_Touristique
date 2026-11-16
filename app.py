import os
import re
import csv
import time
from datetime import datetime

import requests
import torch
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# 0. Réglages généraux
# ---------------------------------------------------------------------------
DATA_FOLDER = "./data"
PERSIST_DIRECTORY = "./chroma_db"

# IMPORTANT : doit être EXACTEMENT le même modèle que dans indexation.py
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

K_CHUNKS = 4          # nombre de morceaux envoyés au modèle de langage
USE_RERANKER = True   # False = pas de reclassement (plus rapide)
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
K_CANDIDATS = 20      # morceaux récupérés avant reclassement
MAX_NEW_TOKENS = 150  # longueur maximale d'une réponse

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TOUTES = "Toutes les villes"   # valeur spéciale : chercher dans tous les fichiers
LOG_FILE = "logs.csv"

if not os.path.exists(PERSIST_DIRECTORY):
    raise FileNotFoundError(
        "La base vectorielle n'existe pas. Veuillez exécuter indexation.py d'abord."
    )

# ---------------------------------------------------------------------------
# 1. Modèle de langage Qwen (chargé EN PREMIER : il a besoin de beaucoup de RAM)
#    GPU en fp16 si disponible, sinon CPU.
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    print("[INFO] GPU détectée :", torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
else:
    print("[INFO] Pas de GPU : le modèle tourne sur le CPU (lent).")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map={"": "cpu"})

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ---------------------------------------------------------------------------
# 2. Base vectorielle ChromaDB + reranker (restent sur le CPU pour économiser la VRAM)
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

reranker = CrossEncoder(RERANKER_MODEL, device="cpu") if USE_RERANKER else None

# ---------------------------------------------------------------------------
# 3. Villes (le nom doit correspondre à <nom>_nettoye.txt dans ./data)
# ---------------------------------------------------------------------------
CAPITALES_COORDS = {
    "Abuja": (9.0579, 7.4951),
    "Addis-Abeba": (9.0300, 38.7400),
    "Paris": (48.8566, 2.3522),
    "Londres": (51.5074, -0.1278),
    "Berlin": (52.5200, 13.4050),
    "Rome": (41.9028, 12.4964),
    "Madrid": (40.4168, -3.7038),
    "Pékin": (39.9042, 116.4074),
    "Tokyo": (35.6762, 139.6503),
    "New Delhi": (28.6139, 77.2090),
    "Riyad": (24.7136, 46.6753),
    "Le Caire": (30.0444, 31.2357),
    "Pretoria": (-25.7479, 28.2293),
    "Rabat": (34.0209, -6.8416),
    "Washington": (38.9072, -77.0369),
    "Ottawa": (45.4215, -75.6972),
    "Brasília": (-15.7975, -47.8919),
    "Brasilia": (-15.7975, -47.8919),
    "Buenos Aires": (-34.6037, -58.3816),
    "Canberra": (-35.2809, 149.1300),
}

# Mots qui désignent une ville sans citer son nom (comparés en minuscules).
# Pas de "France", "Espagne" ou "Maroc" : ces pays correspondent à plusieurs villes.
ALIAS_VILLES = {
    "caire": "Le Caire", "égypte": "Le Caire", "egypte": "Le Caire",
    "japon": "Tokyo",
    "eiffel": "Paris", "louvre": "Paris",
    "big ben": "Londres", "angleterre": "Londres",
    "manhattan": "New York", "statue de la liberté": "New York",
    "beijing": "Pékin", "pekin": "Pékin", "chine": "Pékin",
    "mexique": "Mexico",
    "delhi": "New Delhi", "inde": "New Delhi",
    "arabie": "Riyad",
    "nigeria": "Abuja",
    "éthiopie": "Addis-Abeba", "ethiopie": "Addis-Abeba", "addis abeba": "Addis-Abeba",
    "argentine": "Buenos Aires",
    "brésil": "Brasilia", "bresil": "Brasilia", "brasília": "Brasilia",
    "australie": "Canberra",
    "nouvelle-zélande": "Wellington", "nouvelle zélande": "Wellington",
    "canada": "Ottawa",
    "allemagne": "Berlin",
    "afrique du sud": "Pretoria",
    "maison blanche": "Washington", "maison-blanche": "Washington",
}


def lister_villes_disponibles():
    """Liste les villes qui ont un fichier <Ville>_nettoye.txt dans ./data."""
    if not os.path.exists(DATA_FOLDER):
        return []
    villes = [
        f[: -len("_nettoye.txt")]
        for f in os.listdir(DATA_FOLDER)
        if f.endswith("_nettoye.txt")
    ]
    return sorted(villes)


VILLES_DISPONIBLES = lister_villes_disponibles() or list(CAPITALES_COORDS.keys())

# ---------------------------------------------------------------------------
# 4. Météo (indépendante du chatbot : elle suit seulement la liste déroulante)
# ---------------------------------------------------------------------------
_coords_cache = {}


def trouver_coordonnees(ville_nom):
    """Coordonnées depuis le dictionnaire, sinon via l'API de géocodage Open-Meteo."""
    if ville_nom in CAPITALES_COORDS:
        return CAPITALES_COORDS[ville_nom]
    if ville_nom in _coords_cache:
        return _coords_cache[ville_nom]
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": ville_nom, "count": 1, "language": "fr"},
            timeout=5,
        )
        results = r.json().get("results")
        if results:
            coords = (results[0]["latitude"], results[0]["longitude"])
            _coords_cache[ville_nom] = coords
            return coords
    except Exception:
        pass
    return None


def get_live_weather(ville_nom):
    """Interroge l'API Open-Meteo pour récupérer la météo en direct."""
    coords = trouver_coordonnees(ville_nom)
    if coords is None:
        return "Coordonnées introuvables pour cette ville."

    lat, lon = coords
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        current = data.get("current_weather", {})
        temp = current.get("temperature")
        wind = current.get("windspeed")
        return f"🌡️ Température : {temp}°C | 💨 Vent : {wind} km/h"
    except Exception:
        return "Impossible de récupérer la météo en direct pour le moment."


# ---------------------------------------------------------------------------
# 5. Chatbot : choix de la ville, recherche, génération
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Tu es un guide touristique expert des capitales du monde. "
    "Réponds toujours en français, de façon claire et sympathique. "
    "Réponds en une à trois phrases courtes, en donnant directement l'information demandée, "
    "sans introduction ni répétition de la question. "
    "Base-toi UNIQUEMENT sur le contexte fourni. "
    "Si l'information n'est pas dans le contexte, dis honnêtement que tu ne l'as pas."
)


def contient_mot(texte, mot):
    """Vrai si 'mot' apparaît comme mot entier (évite 'inde' dans 'indépendance')."""
    motif = r"(?<!\w)" + re.escape(mot) + r"(?!\w)"
    return re.search(motif, texte) is not None


def detecter_ville(message, defaut=TOUTES):
    """Trouve la ville visée par la question :
    1) un nom de ville de la base, 2) un alias (ex. 'Eiffel' -> Paris), 3) sinon 'défaut'."""
    msg = message.lower()

    # 1. Nom de ville (les plus longs d'abord : "New Delhi" avant "Delhi")
    for ville in sorted(VILLES_DISPONIBLES, key=len, reverse=True):
        if contient_mot(msg, ville.lower()):
            return ville

    # 2. Alias
    for alias, ville in ALIAS_VILLES.items():
        if ville in VILLES_DISPONIBLES and contient_mot(msg, alias):
            return ville

    # 3. Rien trouvé
    return defaut


def retrieve_context(question, capitale):
    """Récupère les morceaux pertinents : dans une ville précise, ou dans toutes (capitale == TOUTES)."""
    k_recherche = K_CANDIDATS if USE_RERANKER else K_CHUNKS

    if capitale == TOUTES:
        source_path = "toutes les villes"
        docs = vectorstore.similarity_search(question, k=k_recherche)
    else:
        source_path = os.path.join(DATA_FOLDER, f"{capitale}_nettoye.txt")
        docs = vectorstore.similarity_search(
            f"{capitale} : {question}", k=k_recherche, filter={"source": source_path}
        )

    if USE_RERANKER and docs:
        scores = reranker.predict([(question, doc.page_content) for doc in docs])
        classes = sorted(zip(scores, docs), key=lambda paire: paire[0], reverse=True)
        docs = [doc for _, doc in classes][:K_CHUNKS]

    print(f"[DEBUG] ville={capitale} | chunks trouvés={len(docs)} | filtre={source_path}")
    return "\n\n".join(doc.page_content for doc in docs)


def repondre(message, capitale):
    """Cœur du chatbot : renvoie (réponse, contexte). Utilisé aussi par evaluate.py."""
    context = retrieve_context(message, capitale)

    if not context:
        if capitale == TOUTES:
            return "Je n'ai trouvé aucune information correspondant à cette question.", ""
        return (
            f"Je n'ai pas encore de documentation sur {capitale}. "
            "Posez votre question en citant une autre ville de la base."
        ), ""

    entete = "" if capitale == TOUTES else f"Capitale : {capitale}\n\n"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{entete}Contexte :\n{context}\n\nQuestion : {message}",
        },
    ]
    try:
        output = pipe(messages, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return output[0]["generated_text"][-1]["content"].strip(), context
    except Exception as e:
        return f"Une erreur est survenue lors de la génération : {e}", context


# ---------------------------------------------------------------------------
# 6. Journal d'utilisation (logs.csv, pour Power BI)
# ---------------------------------------------------------------------------
MARQUEURS_SANS_INFO = [
    "n'ai pas", "n'ai aucune", "aucune information", "pas d'information",
    "pas spécifiquement mentionné", "je suis désolé",
]


def enregistrer_log(ville, question, reponse, duree, contexte=""):
    """Ajoute une ligne dans logs.csv à chaque question posée dans l'interface."""
    texte = reponse.lower().replace("’", "'")
    sans_info = any(m in texte for m in MARQUEURS_SANS_INFO)
    villes_passages = " / ".join(sorted(set(re.findall(r"Ville : (.+)", contexte))))
    nouveau = not os.path.exists(LOG_FILE)
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            if nouveau:
                w.writerow(["date", "heure", "ville", "question", "duree_s",
                            "sans_info", "reponse", "villes_des_passages", "contexte"])
            maintenant = datetime.now()
            w.writerow([
                maintenant.strftime("%Y-%m-%d"), maintenant.strftime("%H:%M:%S"),
                ville, question, round(duree, 1), sans_info,
                reponse.replace("\n", " "), villes_passages,
                contexte[:300].replace("\n", " | "),
            ])
    except Exception as e:
        print(f"[LOG] impossible d'écrire : {e}")   # le chatbot ne doit jamais planter à cause des logs


def respond(message, history):
    """Fonction déclenchée par Gradio pour chaque message."""
    if history is None:
        history = []

    if not message or not message.strip():
        return "", history

    ville = detecter_ville(message)
    debut = time.time()
    bot_response, contexte = repondre(message, ville)
    enregistrer_log(ville, message, bot_response, time.time() - debut, contexte)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_response})
    return "", history


# ---------------------------------------------------------------------------
# 7. Interface Gradio (la liste déroulante ne sert QU'À la météo)
# ---------------------------------------------------------------------------
with gr.Blocks(title="Chatbot Expert Capitales") as demo:
    gr.Markdown("# 🏙️ Chatbot Expert des Capitales du Monde")
    gr.Markdown(
        "Posez vos questions touristiques. Citez la ville dans votre question "
        "(sinon le chatbot cherche dans toutes les villes)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            ville_meteo = gr.Dropdown(
                choices=VILLES_DISPONIBLES,
                value=VILLES_DISPONIBLES[0],
                label="Ville pour la météo",
            )
            weather_box = gr.Textbox(
                label="Météo en direct",
                value=get_live_weather(VILLES_DISPONIBLES[0]),
                interactive=False,
            )

            ville_meteo.change(
                fn=get_live_weather,
                inputs=[ville_meteo],
                outputs=[weather_box],
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400)
            msg_input = gr.Textbox(
                placeholder="Posez votre question (ex. : Quelle est la population d'Ottawa ?)",
                show_label=False,
            )
            clear_btn = gr.Button("Effacer la conversation")

    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot],
    )
    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()