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

GAME_LOL      = "League of Legends"
GAME_VALORANT = "VALORANT"

print("Fetching latest patch version...")
# Data Dragon requires a patch version in every URL, so we
# fetch the full list and take index [0] which is always
# the most recent patch. This means our data stays current
# automatically every time we run ingestion.
versions = requests.get(
    "https://ddragon.leagueoflegends.com/api/versions.json").json()
latest_patch = versions[0]
print(f"Latest patch version: {latest_patch}")

print("Fetching champion list...")
# Data Dragon splits champion data across two endpoints:
# 1. champion.json: gives us all champion names
# 2. champion/{name}.json : gives us full data per champion
# we need the names first so we can loop through them
# and fetch full data for each one individually.
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
    # splits a large text document into smaller overlapping chunks.
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
    # converts raw champion JSON from Data Dragon into a readable
    # text string that we can embed and store.
    # We deliberately chose these specific fields:
    # - name, title, role: basic identification for any question
    # - lore: covers the lore domain
    # - stats: covers gameplay stat questions
    # - abilities: covers mechanics questions
    # - ally/enemy tips: covers strategy questions
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

def ingest_lol_champions():
    # This is the main LoL ingestion pipeline.
    # For each champion it runs through 4 steps:
    # 1. FETCH — get raw JSON from Data Dragon API
    # 2. FORMAT — convert JSON into readable text
    # 3. CHUNK — split into 400 word pieces with 50 word overlap
    # 4. EMBED + STORE — convert each chunk to a vector and
    #    upsert into Pinecone with metadata
    
    # The metadata we store alongside each vector is critical:
    # - text: the actual content passed to GPT-4o as context
    # - champion: lets us filter by champion name
    # - patch: lets us tell users which patch the data is from
    # - game: lets us separate LoL from VALORANT in search
    # - type: lets us filter by content type (champion vs patch notes)
    # - source: becomes the citation URL in the bot's answer
    
    print(f"\nIngesting {len(champion_names)} LoL champions...")
    for i, champion_name in enumerate(champion_names):
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
                        "game":GAME_LOL,
                        "type": "champion_data",
                        "source": f"https://ddragon.leagueoflegends.com/cdn/{latest_patch}/data/en_US/champion/{champion_name}.json"
                    }
                })
            # use upsert instead of insert so we can safely re-run
            # the script when new patches drop without creating duplicates.
            index.upsert(vectors=vectors)
        except Exception as e:
            print(f"[{i+1}] {champion_name} failed: {e}")
            continue
    print("LoL champion ingestion complete.")


def ingest_valorant_agents():
    # Ingests all playable VALORANT agents from valorant-api.com
    # which is a free community maintained API that mirrors
    # official Riot VALORANT data as clean JSON.
    # We use ?isPlayableCharacter=true to filter out non-playable
    # characters that appear in the API but aren't selectable agents.
    # The structure mirrors our LoL ingestion:
    # FETCH -> FORMAT -> CHUNK -> EMBED + STORE
    # Each agent gets tagged with GAME_VALORANT so search results
    # never mix LoL champion data with VALORANT agent data.
    print("\nIngesting VALORANT agents...")
    try:
        response = requests.get(
            "https://valorant-api.com/v1/agents?isPlayableCharacter=true"
        ).json()
        agents = response["data"]

        for i, agent in enumerate(agents):
            try:
                name= agent["displayName"]
                description = agent["description"]
                role= agent["role"]["displayName"] if agent["role"] else "Unknown"
                
                abilities = "\n".join([
                    f"{a['displayName']}: {a['description']}"
                    for a in agent["abilities"]
                ])
                
                text = f"Agent: {name}, Game: VALORANT, Role: {role}\n"
                text += f"Description: {description}\n"
                text += f"Abilities:\n{abilities}\n"
                text += f"Source: https://playvalorant.com/en-us/agents/{name.lower()}/"
                chunks  = chunk_data(text)
                vectors = []
                for j, chunk in enumerate(chunks):
                    vector = embed_text(chunk)
                    vectors.append({
                        "id": f"valorant-{name}-chunk-{j}",
                        "values": vector,
                        "metadata": {
                            "text": chunk,
                            "agent": name,
                            "game": GAME_VALORANT,
                            "type": "agent_data",
                            "source": f"https://playvalorant.com/en-us/agents/{name.lower()}/"
                        }
                    })
                index.upsert(vectors=vectors)
            except Exception as e:
                print(f"{agent.get('displayName', 'unknown')} failed: {e}")
                continue
    
    except Exception as e:
        print(f"VALORANT ingestion failed: {e}")
    print("VALORANT agent ingestion complete.")


def ingest_esports():
    # Ingests recent esports match results from Leaguepedia
    # using their public MediaWiki CargoQuery API.
    # We use Leaguepedia instead of Riot's official API because
    # Riot's API only covers live game data like match history
    # and ranked stats, it has no endpoint for professional
    # esports tournament results, team rosters, or standings.
    # Leaguepedia is the most comprehensive public database
    # for LoL esports history and exposes it for free.
    # We limit to 100 matches to keep the data current and 
    # the chunk count manageable.
    # All matches get stored as one document and chunked together
    # so the bot can answer questions like "who won Worlds 2024"
    # or "what are the current LCK standings".
    print("\nIngesting esports data from Leaguepedia...")
    try:
        url = "https://lol.fandom.com/api.php"
        params = {
            "action": "cargoquery",
            "format": "json",
            "tables": "ScoreboardGames",
            "fields": "Team1, Team2, WinTeam, DateTime_UTC, Tournament",
            "where": "DateTime_UTC > '2024-01-01'",
            "limit": "100",
            "order_by": "DateTime_UTC DESC"
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Leaguepedia API returned {response.status_code}")
            return
        matches = response.json().get("cargoquery", [])

        if not matches:
            print("No esports data returned")
            return

        text = "Recent Esports Results for League of Legends 2024\n\n"
        for match in matches:
            m = match["title"]
            tournament = m.get("Tournament", "")
            team1 = m.get("Team1", "")
            team2 = m.get("Team2", "")
            winner = m.get("Winner", "")
            date = m.get("DateTime UTC", "")
            text += f"{tournament}: {team1} vs {team2} — Winner: {winner} ({date})\n"

        text += "\nSource: https://lol.fandom.com/wiki/League_of_Legends_Esports_Wiki"

        chunks = chunk_data(text)
        vectors = []

        for j, chunk in enumerate(chunks):
            vector = embed_text(chunk)
            vectors.append({
                "id": f"esports-2024-chunk-{j}",
                "values": vector,
                "metadata": {
                    "text": chunk,
                    "game": GAME_LOL,
                    "type": "esports",
                    "source": "https://lol.fandom.com/wiki/League_of_Legends_Esports_Wiki"
                }
            })

        index.upsert(vectors=vectors)
        print(f"{len(matches)} esports matches and {len(chunks)} chunks")

    except Exception as e:
        print(f"Esports ingestion failed: {e}")

    print("Esports ingestion complete.")


ingest_lol_champions()
ingest_valorant_agents()
ingest_esports()

print("\nAll ingestion complete.")
print("Check Pinecone dashboard for final record count.")
