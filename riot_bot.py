from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


FAKE_CONTEXT = """
--- DOCUMENT 1 ---
Patch 14.10 Notes (League of Legends):
Jinx: Base attack damage reduced from 57 to 52.
Jinx: Q (Switcheroo!) attack speed bonus increased by 5%.
Jinx: W (Zap!) mana cost reduced from 50 to 40 at all ranks.
Source: https://lolesports.com/en-US/news/patch-14-10-notes

--- DOCUMENT 2 ---
Jinx champion overview:
Jinx is a marksman who excels at long-range teamfight damage.
Her passive 'Get Excited!' grants massive attack speed and
movement speed when she scores a kill or assist.
Source: https://leagueoflegends.com/en-us/champions/jinx/
"""

history = []

# this handles the memory of the conversation. When responding to a new message,
# it includes additional context from the conversation history which is necessary 
# for maintaining context but can provide redundant information.

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit" or user_input.lower() == "exit":
        break
    
    if not user_input:
        continue
    
    # store user's message in history for reference
    history.append({"role": "user", "content": user_input})
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nCONTEXT:\n" + FAKE_CONTEXT},
        *history # this will include all previous messages so the chatbot can maintain context while responding
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    answer = response.choices[0].message.content
    print("Bot:", answer)