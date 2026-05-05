from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

SYSTEM_PROMPT = """
You are a Riot Games assistant. You help players with questions
about League of Legends, VALORANT, Teamfight Tactics, and esports.
You cover patch notes, champion and agent lore, player and team
esports history, game mechanics, and champion or agent abilities.

Rules you must ALWAYS follow:

1. Only answer using the CONTEXT provided to you below.
2. Never use your own training memory to answer.
3. Every answer must end with: [Source: URL or document name]
4. If the context does not contain the answer, say:
   'I don't have a verified source for this. Please check
   lolesports.com or support.riotgames.com for accurate info.'
5. Never guess. Never make up stats, patch numbers, or player names.
6. Always identify which Riot game you are answering about.
   Never mix information between games.
7. Always state which patch version your information is from.
   If you cannot confirm the patch version from context, say so.
8. Do not make predictions or give opinions on future events,
   player rankings, or competitive outcomes. These cannot be cited.
9. For numerical stats, quote exact numbers from the source.
   Do not round, estimate, or generalize.
"""

def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def search_pinecone(query, top_k=5):
    # search for the most relevant documents in Pinecone top 5 set by default
    query_vector = embed_text(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    # index.query() returns a dictionary (list) of the most similar chunks, 
    # sorted by score from highest to lowest inside the "matches" key
    return results["matches"]

def build_context(matches):
    context = ""
    for i, match in enumerate(matches):
        text   = match["metadata"]["text"]
        source = match["metadata"]["source"]
        score  = match["score"]
        context += f"Document {i+1} has score {score}\n"
        context += f"Text: {text}\n"
        context += f"Source: {source}\n\n"
    return context

def rewrite_query(user_input, history):
    # this handles rewriting the user's query for better search results
    if not history:
        return user_input
    
    messages = [
        {
            "role": "system",
            "content": """You are a search query optimizer.
            Given a conversation history and a new user question, rewrite the 
            question as a clear standalone search query resolving any pronouns 
            or references to specific names.
            Return ONLY the rewritten query. Nothing else."""
        },
        *history,
        {
            "role": "user",
            "content": f"Rewrite as standalone search query: {user_input}"
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=50
    )
    
    rewritten = response.choices[0].message.content
    print(f"Rewritten User input: '{user_input}' to query '{rewritten}'")
    return rewritten

def evaluate_answer(question, answer, context):
    # sends a second LLM call to check whether the answer
    # is actually grounded in the retrieved context.
    # this is the "LLM as judge" evaluation pattern —
    # we use a cheap fast model to verify the main model's output.
    # returns True if grounded, False if hallucination detected.
    if not context:
        return True  # no context means bot already refused — that's correct behavior
    
    messages = [
        {
            "role": "system",
            "content": """You are an answer evaluator.
            Given a question, an answer, and source context documents,
            determine if the answer is fully supported by the context.
            Reply with only: GROUNDED or HALLUCINATION
            GROUNDED means every claim in the answer appears in the context.
            HALLUCINATION means the answer contains claims not in the context."""
        },
        {
            "role": "user",
            "content": f"""
                Question: {question}
                Answer: {answer}
                Context: {context}
                Is the answer grounded in the context? Reply GROUNDED or HALLUCINATION only.
                """
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=10
    )

    verdict = response.choices[0].message.content.strip().upper()
    return verdict == "GROUNDED"

print("Hi, I am a Riot Games assistant. Ask me questions about games, esports, players, patches, lore, and more.")
print("Type quit or exit to end the conversation.")

history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit" or user_input.lower() == "exit":
        break
    
    if not user_input:
        continue
    
    # search pinecone for relevant chunks
    search_query = rewrite_query(user_input, history)
    matches = search_pinecone(search_query)
    # build the context from the search results
    context = build_context(matches)
    
    history.append({"role": "user", "content": user_input})
    
    # the message to gpt include the current user input and the context
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\nCONTEXT:\n" + context
        },
        *history
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    answer = response.choices[0].message.content
    
    is_grounded = evaluate_answer(user_input, answer, context)
    if not is_grounded:
        print("Evaluation:   Answer may contain information not in verified sources. This may be a hallucination.")
    
    history.append({"role": "assistant", "content": answer})
    
    print(f"\nRiotBot: {answer}\n")