# 🍽️ GoodFoods AI Reservation Agent  
**Intelligent Restaurant Recommendation & Reservation System (FDSE Assignment Project)**  

This project is an end-to-end **AI-powered reservation assistant** designed for GoodFoods — a multi-location restaurant chain.  
The system automatically recommends restaurants, checks availability, books tables, and retrieves bookings through natural language conversation.

It uses:
- **Streamlit** frontend  
- **SQLite** local database  
- **OpenRouter LLM (Grok 4.1 Fast)** with **tool-calling**  
- **Python tool functions** (recommend, create reservation, availability check, show bookings)

---

## 🚀 Features

### ✔ Conversational Interface  
Natural language chat interface powered by an LLM.

### ✔ Smart Recommendations  
Suggest restaurants based on:
- cuisine  
- location  
- party size  
- time/date  
- budget (optional)

### ✔ Real-Time Availability Check  
Ensures capacity and overbooking protection.

### ✔ Instant Reservations  
Creates confirmed bookings stored in SQLite.

### ✔ Booking Lookup  
Retrieve bookings using phone number.

### ✔ Complete Tool-Calling Architecture  
LLM *decides* when to call:
- `recommend`
- `check_availability`
- `create_reservation`
- `show_bookings`

### ✔ Fallback Post-Processing  
After tool execution, the LLM generates a **friendly human message** summarizing the result.

---

## 🏗️ Project Structure

goodfoods-ai-agent/
│
├── app.py # Streamlit UI + tool handling + LLM integration
├── restaurants.csv # 100 generated restaurant entries
├── generate_restaurants.py # Script to generate fake data
├── goodfoods.db # SQLite DB (auto-created)
├── prompt/
│ └── few_shots.txt # (Optional) Few-shot examples for the agent
├── README.md # Documentation
└── requirements.txt # Python dependencies


---

## 🔧 Installation & Setup

### 1. Clone the repository  

git clone https://github.com/<your-username>/goodfoods-ai-agent.git
cd goodfoods-ai-agent

python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

### 2. Set your OpenRouter API key


setx OPENROUTER_API_KEY "your_key_here"
Then restart your terminal.

Verify:

python - <<EOF
import os
print(os.getenv("OPENROUTER_API_KEY"))
EOF
▶️ Running the Application


streamlit run app.py
Access the app at:

Local URL: http://localhost:8501

Network URL: shown by Streamlit


## 💬 Example Conversations

Restaurant Recommendation

User: recommend a place for 2 people italian in bandra tonight at 8pm
AI: suggests top 3 matches from DB

Availability Check

User: Is table available at restaurant id 64 on 2025-11-25 at 19:00 for 2?

Reservation

User: Book at id 64 for 2 people on 2025-11-25 19:00 name: Ravi phone: 9999999999
AI: Reservation confirmed! ID: 1

Show Bookings

User enters phone on left panel:
Shows all bookings stored in the database.

## 🧠 Prompt Engineering Notes

The system prompt contains:

Tool schemas

Strict JSON requirements

Error handling

Behavioral instructions

Few-shot examples help the LLM consistently output tool calls.

A pre-parser extracts strong signals:

cuisine

location

date

time

party size

This dramatically improves both tool accuracy and natural language flexibility.