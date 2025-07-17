import streamlit as st
import json
from pathlib import Path
import openai
from dotenv import dotenv_values
#from openai import OpenAI

model_pricings = {
    "gpt-4o": {
        "input_tokens": 5.00 / 1_000_000, # per token
        "output_tokens": 15.00 / 1_000_000, # per token 
    },
    "gpt-4o-mini": {
        "input_tokens": 0.150 / 1_000_000, # per token
        "output_tokens": 0.600 / 1_000_000, # per token 
    }
}

# Wczytywanie zmiennych środowiskowych z pliku .env
env = dotenv_values(".env")

MODEL = "gpt-4o-mini"
USD_TO_PLN = 3.97
PRICING = model_pricings[MODEL]


# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.title("Zaloguj do OpenAI")
        
        # Dodanie instrukcji dla użytkownika z kolorową ramką
        instruction_html = """
        <div style="background-color: #f0f4f8; padding: 10px; border-radius: 5px; border: 1px solid #0073e6; margin-bottom: 10px;">
            <h4>Instrukcje uzyskania klucza API</h4>
            <ol>
                <li>Załóż konto na stronie <a href="https://platform.openai.com/signup" target="_blank">OpenAI</a>.</li>
                <li>Wygeneruj swój klucz API w sekcji API Keys.</li>
                <li>Wklej go poniżej.</li>
            </ol>
        </div>
        """
        st.markdown(instruction_html, unsafe_allow_html=True)
        
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

openai.api_key = st.session_state["openai_api_key"]

def get_chatbot_reply(user_prompt, memory):
    # dodaj system message
    messages = [
        {
            "role": "system",
            "content": st.session_state["chatbot_personality"],
        },
    ]

    # dodaj wszystkie wiadomości z pamięci
    for message in memory:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    # dodaj wiadomość użytkownika
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    try:
        # Użycie OpenAI API
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages
        )

        # Sprawdzanie struktury response
        usage = {
            "prompt_tokens": response['usage']['prompt_tokens'],
            "completion_tokens": response['usage']['completion_tokens'],
            "total_tokens": response['usage']['total_tokens'],
        }

        return {
            "role": "assistant",
            "content": response.choices[0].message['content'],  # Użycie ['content'] do uzyskania treści
            "usage": usage,
        }
    except Exception as e:
        st.error(f"Błąd podczas uzyskiwania odpowiedzi: {e}")
        return {"role": "assistant", "content": "Coś poszło nie tak. Sprawdź klucz API lub połączenie."}
#
# CONVERSATION HISTORY AND DATABASE
#
DEFAULT_PERSONALITY = """
Jesteś pomocnikiem, który odpowiada na wszystkie pytania użytkownika.
Odpowiadaj na pytania w sposób zwięzły i zrozumiały.
""".strip()

DB_PATH = Path("db")
DB_CONVERSATIONS_PATH = DB_PATH / "conversation"

def load_conversation_to_state(conversation):
    st.session_state["id"] = conversation["id"]
    st.session_state["name"] = conversation["name"]
    st.session_state["messages"] = conversation["messages"]
    st.session_state["chatbot_personality"] = conversation["chatbot_personality"]

def load_current_conversation():
    if not DB_PATH.exists():
        DB_PATH.mkdir()
        DB_CONVERSATIONS_PATH.mkdir()
        conversation_id = 1
        conversation = {
            "id": conversation_id,
            "name": "Konwersacja 1",
            "chatbot_personality": DEFAULT_PERSONALITY,
            "messages": [],
        }

        # tworzymy nową konwersację
        with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
            f.write(json.dumps(conversation))

        # która od razu staje się aktualna
        with open(DB_PATH / "current.json", "w") as f:
            f.write(json.dumps({
                "current_conversation_id": conversation_id,
            }))

    else:
        # sprawdzamy, która konwersacja jest aktualna
        with open(DB_PATH / "current.json", "r") as f:
            data = json.loads(f.read())
            conversation_id = data["current_conversation_id"]

        # wczytujemy konwersację
        with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
            conversation = json.loads(f.read()) 

    load_conversation_to_state(conversation)

def save_current_conversation_message():
    conversation_id = st.session_state["id"]
    new_messages = st.session_state["messages"]

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "messages": new_messages,
        }))

def save_current_conversation_name():
    conversation_id = st.session_state["id"]
    new_conversation_name = st.session_state["new_conversation_name"]

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "name": new_conversation_name,
        }))

def save_current_conversation_personality():
    conversation_id = st.session_state["id"]
    new_chatbot_personality = st.session_state["new_chatbot_personality"]

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps({
            **conversation,
            "chatbot_personality": new_chatbot_personality,
        }))

def create_new_conversation():
    conversation_ids = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        conversation_ids.append(int(p.stem))

    conversation_id = max(conversation_ids) + 1
    personality = DEFAULT_PERSONALITY
    if "chatbot_personality" in st.session_state and st.session_state["chatbot_personality"]:
        personality = st.session_state["chatbot_personality"]

    conversation = {
        "id": conversation_id,
        "name": f"Konwersacja {conversation_id}",
        "chatbot_personality": personality,
        "messages": [],
    }

    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "w") as f:
        f.write(json.dumps(conversation))

    with open(DB_PATH / "current.json", "w") as f:
        f.write(json.dumps({
            "current_conversation_id": conversation_id,
        }))

    load_conversation_to_state(conversation)
    st.rerun()

def switch_conversation(conversation_id):
    with open(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", "r") as f:
        conversation = json.loads(f.read())

    with open(DB_PATH / "current.json", "w") as f:
        f.write(json.dumps({
            "current_conversation_id": conversation_id,
        }))

    load_conversation_to_state(conversation)
    st.rerun()

def list_conversations():
    conversations = []
    for p in DB_CONVERSATIONS_PATH.glob("*.json"):
        with open(p, "r") as f:
            conversation = json.loads(f.read())
            conversations.append({
                "id": conversation["id"],
                "name": conversation["name"]
            })

    return conversations

#
# MAIN PROGRAM
#
st.title('Chatbot')
load_current_conversation()

# Wyświetlanie wiadomości
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input('O co chcesz spytać?')

if prompt:
    with st.chat_message("human"):
        st.markdown(prompt)

    st.session_state["messages"].append({
        "role": "user",
        "content": prompt
    })

    chatbot_message = get_chatbot_reply(
        prompt,
        memory=st.session_state["messages"][-20:]
    )

    with st.chat_message("assistant"):
        st.markdown(chatbot_message["content"])

    st.session_state["messages"].append(chatbot_message)
    save_current_conversation_message()

with st.sidebar:
    st.write("Aktualny model", MODEL)
    total_cost = 0
    for message in st.session_state["messages"]:
        if "usage" in message:
            total_cost += message["usage"]["prompt_tokens"] * PRICING["input_tokens"]
            total_cost += message["usage"]["completion_tokens"] * PRICING["output_tokens"]

    c0, c1 = st.columns(2)
    with c0:
        st.metric("Koszt rozmowy (USD)", f"${total_cost:.4f}")

    with c1:
        st.metric("Koszt rozmowy (PLN)", f"{total_cost * USD_TO_PLN:.4f}")

    st.session_state["name"] = st.text_input(
        "Nazwa konwersacji",
        value=st.session_state["name"],
        key="new_conversation_name",
        on_change=save_current_conversation_name,
    )

    st.session_state["chatbot_personality"] = st.text_area(
        "Osobowość czatbota",
        max_chars=1000,
        height=200,
        value=st.session_state["chatbot_personality"],
        key="new_chatbot_personality",
        on_change=save_current_conversation_personality,
    )  

    st.subheader("Konwersacje")
    if st.button("Nowa konwersacja"):
        create_new_conversation()

    # pokazujemy tylko top 5 konwersacji
    conversations = list_conversations()
    sorted_conversations = sorted(conversations, key=lambda x: x["id"], reverse=True)
    for conversation in sorted_conversations[:5]:
        c0, c1 = st.columns([10, 4])
        with c0:
            st.write(conversation["name"])

        with c1:
            if st.button("załaduj", key=conversation["id"], disabled=conversation["id"] == st.session_state["id"]):
                switch_conversation(conversation["id"])