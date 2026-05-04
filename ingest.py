from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import requests
import os
import time

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

print("Fetching latest patch version...")
versions = requests.get(
    "https://ddragon.leagueoflegends.com/api/versions.json").json()
latest_patch = versions[0]
print(f"Latest patch version: {latest_patch}")

print("Fetching champion list...")
champion_response = requests.get(
    f"https://ddragon.leagueoflegends.com/cdn/{latest_patch}/data/en_US/champion.json").json()
champion_names = list(champion_response["data"].keys())
print(f"Found {len(champion_names)} champions")

def get_champion_data(name, patch):
    # this returns data for one specific champion from data dragon
    print(f"Fetching data for {name}...")
    url = f"https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion/{name}.json"
    response = requests.get(url).json()
    return response["data"][name]

def chunk_data(text, chunk_size= 400, overlap=50):
    # overlap 50 words means if the chunk size is 400 the next chunk will start at word 350
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        end = i + chunk_size
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
    return chunks

def embed_text(text):
    # convert text to embeddings using OpenAI's embedding API
    response = client.embeddings.create(
        input= text,
        model= "text-embedding-3-small"
    )
    return response.data[0].embedding

def format_champion_text(champion_data, patch):
    # format the champion data (json) into a text string
    name = champion_data["name"]
    title = champion_data["title"]
    lore = champion_data["lore"]
    tags = ", ".join(champion_data["tags"])
    hp = champion_data["stats"]["hp"]
    ad = champion_data["stats"]["attackdamage"]
    armor = champion_data["stats"]["armor"]
    movespeed = champion_data["stats"]["movespeed"]
    mp = champion_data["stats"]["mp"]
    attackrange = champion_data["stats"]["attackrange"]
    ally_tips = " ".join(champion_data["allytips"])
    enemy_tips = " ".join(champion_data["enemytips"])
    
    spells = []
    for spell in champion_data["spells"]:
        spells.append(f"{spell['id']}, {spell['name']}: {spell['description']}")
    abilities = "\n".join(spells)

    text = f"Champion: {name}, Title: {title}, Patch: {patch}, Role: {tags}\n"
    text += f"Lore: {lore}\n"
    text += f"Stats: HP: {hp}, AD: {ad}, Armor: {armor}, Move Speed: {movespeed}, MP: {mp}, Attack Range: {attackrange}\n"
    text += f"Abilities:\n{abilities}\n"
    text += f"Tips for playing as {name}: {ally_tips}\n"
    text += f"Tips for playing against {name}: {enemy_tips}\n"
    text += f"Source: https://ddragon.leagueoflegends.com/cdn/{patch}/data/en_US/champion/{name}.json"

    return text

champions_to_ingest = champion_names
print(f"Preparing to ingest data for {len(champions_to_ingest)} champions.")

for i, champion_name in enumerate(champions_to_ingest):
    try:
        champion_data = get_champion_data(champion_name, latest_patch)
        text = format_champion_text(champion_data, latest_patch)
        chunks = chunk_data(text)
        vectors = []
        for j, chunk in enumerate(chunks):
            vector = embed_text(chunk)
            vectors.append({
                "id": f"{champion_name}-chunk-{j}",
                "values": vector,
                "metadata": {
                    "text": chunk,
                    "champion": champion_name,
                    "patch": latest_patch,
                    "game": "League of Legends",
                    "type": "champion_data",
                    "source": f"https://ddragon.leagueoflegends.com/cdn/{latest_patch}/data/en_US/champion/{champion_name}.json"
                }
            })
        index.upsert(vectors) # upsert uploads the vectors into the Pinecone index
    except Exception as e:
        print(f"Error processing {champion_names}: {e}")
    
print("\nIngestion complete.")
print("Check the Pinecone index for the ingested data.")