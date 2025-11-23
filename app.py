# app.py
"""
GoodFoods Reservation Agent (updated)
- Improved natural-language date/time parsing and intent extraction
- Slightly modernized Streamlit chat UI (CSS chat bubbles + sidebar controls)
- Retains OpenRouter (x-ai/grok-4.1-fast) integration and tool-calling flow
"""

import os
import streamlit as st
import sqlite3
import json
import re
import time
from datetime import datetime, timedelta
import pandas as pd
import requests

# ---------- Configuration ----------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLAMA_MODEL = "x-ai/grok-4.1-fast"  # discovered working model
DB = "goodfoods.db"
#LLAMA_MODEL = "meta-llama/llama-3.3-8b-instruct"

# ---------- DB init ----------
def init_db():
    conn = sqlite3.connect(DB, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS restaurants (
                 id INTEGER PRIMARY KEY, name TEXT, cuisine TEXT, capacity INTEGER, area TEXT,
                 price_bucket TEXT, rating REAL, amenities TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS reservations (
                 id INTEGER PRIMARY KEY AUTOINCREMENT, restaurant_id INTEGER, datetime TEXT,
                 party_size INTEGER, name TEXT, phone TEXT, email TEXT, status TEXT, created_at TEXT)""")
    conn.commit()
    return conn

conn = init_db()

# load CSV into db if empty
def load_csv_to_db(csvfile="restaurants.csv"):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM restaurants")
    if c.fetchone()[0] == 0:
        df = pd.read_csv(csvfile)
        df.to_sql("restaurants", conn, if_exists="append", index=False)

load_csv_to_db()

# ---------- Tools (local) ----------
def tool_recommend(args):
    party_size = int(args.get("party_size", 2) or 2)
    cuisine = args.get("cuisine")
    area = args.get("location") or args.get("area")
    price = args.get("budget") or args.get("price_bucket")

    if cuisine:
        cuisine = str(cuisine).strip().lower()
    if area:
        area = str(area).strip().lower()
    if price:
        price = str(price).strip().lower()

    q = "SELECT id,name,cuisine,capacity,area,price_bucket,rating,amenities FROM restaurants WHERE capacity>=? "
    params = [party_size]
    if cuisine:
        q += "AND LOWER(cuisine) LIKE ? "
        params.append(f"%{cuisine}%")
    if area:
        q += "AND LOWER(area) LIKE ? "
        params.append(f"%{area}%")
    if price:
        q += "AND LOWER(price_bucket)=? "
        params.append(price)
    q += "ORDER BY rating DESC LIMIT 10"

    print("DEBUG tool_recommend SQL:", q)
    print("DEBUG params:", params)

    cur = conn.execute(q, params)
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]

    matched = bool(rows)
    if not rows:
        q2 = "SELECT id,name,cuisine,capacity,area,price_bucket,rating,amenities FROM restaurants WHERE capacity>=? ORDER BY rating DESC LIMIT 3"
        cur2 = conn.execute(q2, (party_size,))
        rows = [dict(zip([c[0] for c in cur2.description], r)) for r in cur2.fetchall()]

    return {"candidates": rows[:3], "matched": matched, "filters": {"party_size": party_size, "cuisine": cuisine, "area": area, "price": price}}

def tool_check_availability(args):
    try:
        restaurant_id = int(args["restaurant_id"])
        date = args["date"]
        time_s = args["time"]
        party_size = int(args["party_size"])
    except Exception as e:
        return {"available": False, "reason": "invalid_args", "error": str(e)}

    dt = f"{date} {time_s}"
    cur = conn.cursor()
    cur.execute("SELECT capacity FROM restaurants WHERE id=?", (restaurant_id,))
    row = cur.fetchone()
    if not row:
        return {"available": False, "reason": "restaurant_not_found"}
    capacity = row[0]
    cur.execute("SELECT SUM(party_size) FROM reservations WHERE restaurant_id=? AND datetime=?", (restaurant_id, dt))
    used = cur.fetchone()[0] or 0
    available = (used + party_size) <= capacity
    return {"available": available, "used": used, "capacity": capacity}

def tool_create_reservation(args):
    try:
        restaurant_id = int(args["restaurant_id"])
        date = args["date"]
        time_s = args["time"]
        party_size = int(args["party_size"])
        name = args["name"]
    except Exception as e:
        return {"success": False, "reason": "invalid_args", "error": str(e)}

    dt = f"{date} {time_s}"
    phone = args.get("phone", "")
    email = args.get("email", "")
    created_at = datetime.utcnow().isoformat()
    cur = conn.cursor()
    av = tool_check_availability({"restaurant_id": restaurant_id, "date": date, "time": time_s, "party_size": party_size})
    if not av.get("available", False):
        return {"success": False, "reason": "not_available", "details": av}
    cur.execute("INSERT INTO reservations (restaurant_id,datetime,party_size,name,phone,email,status,created_at) VALUES (?,?,?,?,?,?,?,?)",
               (restaurant_id, dt, party_size, name, phone, email, "CONFIRMED", created_at))
    conn.commit()
    return {"success": True, "reservation_id": cur.lastrowid}

def tool_show_bookings(args):
    phone = args.get("phone")
    if not phone:
        return {"bookings": []}
    cur = conn.cursor()
    cur.execute("SELECT r.id, rest.name, r.datetime, r.party_size, r.status FROM reservations r JOIN restaurants rest ON rest.id=r.restaurant_id WHERE phone=?", (phone,))
    rows = cur.fetchall()
    return {"bookings": [dict(id=r[0],restaurant=r[1],datetime=r[2],party_size=r[3],status=r[4]) for r in rows]}


# ---------- LLM integration (OPENROUTER) ----------
SYSTEM_PROMPT = """
You are GoodFoods Reservation Assistant. Decide whether the user request requires a tool call (recommend/check_availability/create_reservation/show_bookings) or a plain conversational reply.
If you decide a tool call is needed, respond with EXACTLY a JSON object (and nothing else) using this schema:
{"tool":"<tool_name>", "args":{...}}
Allowed tools:
- recommend: args = { "party_size": int, "date": "YYYY-MM-DD" (optional), "time":"HH:MM" (optional), "cuisine": str (optional), "location": str (optional), "budget": str (optional) }
- check_availability: args = { "restaurant_id": int, "date":"YYYY-MM-DD", "time":"HH:MM", "party_size": int }
- create_reservation: args = { "restaurant_id": int, "date":"YYYY-MM-DD", "time":"HH:MM", "party_size": int, "name": str, "phone": str, "email": str (optional) }
- show_bookings: args = { "phone": str }
When returning JSON, do not include any extra text, quotes, or code fences.
If no tool is required, reply in plain text.
"""

FEW_SHOT_EXAMPLES = [
    {"role":"user","content":"Suggest an Italian place for 4 in Bandra tonight at 20:00"},
    {"role":"assistant","content":'{"tool":"recommend","args":{"party_size":4,"date":"2025-11-21","time":"20:00","cuisine":"Italian","location":"Bandra"}}'},
    {"role":"user","content":"Is table available at restaurant id 12 on 2025-11-24 19:00 for 3?"},
    {"role":"assistant","content":'{"tool":"check_availability","args":{"restaurant_id":12,"date":"2025-11-24","time":"19:00","party_size":3}}'},
    {"role":"user","content":"Book at id 12 for 3 on 2025-11-24 19:00 name: Asha phone: 9999999999"},
    {"role":"assistant","content":'{"tool":"create_reservation","args":{"restaurant_id":12,"date":"2025-11-24","time":"19:00","party_size":3,"name":"Asha","phone":"9999999999"}}'}
]

# ---------- Helper: robust date/time parsing ----------
WEEKDAY_MAP = {
    "monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6
}

def parse_time_fragment(text):
    """Return HH:MM or None. Handles '8pm', '8:30 pm', '20:00', '7.30', 'at 8'."""
    text = text.strip().lower()
    # common keywords
    if "dinner" in text:
        return "19:00"
    if "lunch" in text:
        return "13:00"
    if "breakfast" in text:
        return "09:00"
    # regex for 'in N hours'
    m = re.search(r'in\s+(\d+)\s*hours?', text)
    if m:
        delta = int(m.group(1))
        t = datetime.now() + timedelta(hours=delta)
        return t.strftime("%H:%M")
    # find patterns like 8pm, 8:30pm, 20:00
    m = re.search(r'(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?', text)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        if ampm:
            if ampm == "pm" and h != 12:
                h += 12
            if ampm == "am" and h == 12:
                h = 0
        # clamp
        h = h % 24
        return f"{h:02d}:{mm:02d}"
    return None

def parse_date_fragment(text):
    """Return YYYY-MM-DD or None. Handles today/tomorrow/tonight and weekday names and explicit dates YYYY-MM-DD"""
    text = text.strip().lower()
    today = datetime.now().date()
    if "today" in text:
        return today.isoformat()
    if "tonight" in text:
        return today.isoformat()
    if "tomorrow" in text:
        return (today + timedelta(days=1)).isoformat()
    # ISO date
    m = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m:
        return m.group(1)
    # weekday names: "on friday" or "next fri"
    for name, idx in WEEKDAY_MAP.items():
        if name[:3] in text or name in text:
            # compute next weekday occurrence (including today if same day and time later)
            today_idx = today.weekday()
            delta = (idx - today_idx) % 7
            if "next" in text and delta == 0:
                delta = 7
            return (today + timedelta(days=delta)).isoformat()
    return None

def parse_date_time(user_text):
    """
    Attempt to extract date and time from arbitrary user_text.
    Returns dict with optional 'date' (YYYY-MM-DD) and 'time' (HH:MM).
    """
    out = {}
    # look for explicit time tokens
    time_val = parse_time_fragment(user_text)
    date_val = parse_date_fragment(user_text)

    # If 'tonight' present but no explicit time, default dinner time
    if "tonight" in user_text and not time_val:
        time_val = "20:00"

    # If 'this evening' or 'this weekend' basic heuristics
    if "evening" in user_text and not time_val:
        time_val = "20:00"

    # If user says 'for 2 people tomorrow at 8' both fragments will be found
    if date_val:
        out["date"] = date_val
    if time_val:
        out["time"] = time_val

    return out

# ---------- OpenRouter caller (robust, uses discover_model) ----------
# reuse discover_model/caller code from your working file (kept minimal here)
def fetch_openrouter_models():
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("fetch_openrouter_models error:", e)
        return None

_cached_model_name = None
_last_model_fetch = 0

def discover_model(preferred=LLAMA_MODEL):
    global _cached_model_name, _last_model_fetch
    if _cached_model_name and (time.time() - _last_model_fetch) < 300:
        return _cached_model_name
    # probe preferred quickly
    probe_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    probe_payload = {"model": preferred, "messages":[{"role":"system","content":"probe"},{"role":"user","content":"hi"}], "max_tokens":1}
    try:
        r = requests.post(probe_url, headers=headers, json=probe_payload, timeout=10)
        if r.status_code == 200:
            _cached_model_name = preferred
            _last_model_fetch = time.time()
            return _cached_model_name
    except Exception:
        pass
    # fallback to list
    models_resp = fetch_openrouter_models()
    _last_model_fetch = time.time()
    if not models_resp:
        return preferred
    model_entries = models_resp.get("data") if isinstance(models_resp, dict) else models_resp
    candidates = []
    if isinstance(model_entries, list):
        for m in model_entries:
            name = None
            if isinstance(m, dict):
                name = m.get("id") or m.get("name")
            elif isinstance(m, str):
                name = m
            if not name:
                continue
            low = name.lower()
            if "grok" in low or "gpt" in low or "gemini" in low or "llama" in low:
                candidates.append(name)
    if candidates:
        _cached_model_name = candidates[0]
        return _cached_model_name
    return preferred

def call_llm_openrouter(conversation_text, temperature=0.0, max_tokens=512):
    if not OPENROUTER_API_KEY:
        return "ERROR_CALLING_LLM: OPENROUTER_API_KEY not set."
    model_to_use = discover_model(preferred=LLAMA_MODEL)
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role":"user","content":conversation_text})
    payload = {"model": model_to_use, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as e:
        return f"ERROR_CALLING_LLM: network error: {e}"
    status = r.status_code
    try:
        data = r.json()
    except Exception:
        return f"ERROR_CALLING_LLM: non-json response status={status} body={r.text[:1000]}"
    # debug: show JSON in sidebar
    try:
        st.sidebar.write("OpenRouter JSON:", data)
        st.sidebar.write("Using model:", model_to_use)
    except Exception:
        pass
    if status != 200:
        err_detail = data.get("error") or data
        return f"ERROR_CALLING_LLM: status={status} detail={json.dumps(err_detail)[:1000]}"
    # extract text
    if isinstance(data, dict) and "choices" in data and len(data["choices"])>0:
        choice = data["choices"][0]
        # message.content or text
        if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
            return choice["message"]["content"]
        if isinstance(choice, dict) and "text" in choice:
            return choice["text"]
    # fallback: find first string
    def find_first_string(obj):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                s = find_first_string(v)
                if s:
                    return s
        if isinstance(obj, list):
            for item in obj:
                s = find_first_string(item)
                if s:
                    return s
        return None
    fallback = find_first_string(data)
    if fallback:
        return fallback
    return "ERROR_CALLING_LLM: Unexpected response shape. See OpenRouter JSON in sidebar."

def safe_parse_json(text):
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return None
        candidate = text[start:end+1]
        return json.loads(candidate)
    except Exception:
        return None

# ---------- Streamlit UI (modernized) ----------
st.set_page_config(page_title="GoodFoods Reservation Agent", layout="wide")
# small CSS for chat bubbles
st.markdown("""
<style>
.chat-box { max-height: 540px; overflow:auto; padding:10px; background:#f7f7f9; border-radius:8px; }
.user { background:#0B5FFF;color:white;padding:10px;border-radius:12px; margin:6px 0; display:inline-block; }
.agent { background:#eef2ff;color:#0b1b3f;padding:10px;border-radius:12px; margin:6px 0; display:inline-block; }
.meta { color:#777; font-size:12px; margin-bottom:6px; }
.sidebar-section { padding:8px 0; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,2])

with col1:
    st.title("GoodFoods")
    st.subheader("Restaurants")
    df = pd.read_sql("SELECT id,name,cuisine,capacity,area,price_bucket,rating FROM restaurants LIMIT 50", conn)
    st.dataframe(df, use_container_width=True)
    st.markdown("**Quick actions**")
    lookup_phone = st.text_input("Enter phone for lookup (left)", value="", key="lookup_phone")
    if st.button("Show my bookings (demo)", key="show_bookings_btn"):
        if lookup_phone.strip():
            out = tool_show_bookings({"phone":lookup_phone.strip()})
            st.write(out)
        else:
            st.warning("Enter a phone number to show bookings.")
    if st.button("Clear conversation"):
        st.session_state.history = []

    # Hidden agent settings (no UI controls shown)
    st.markdown("---")
    st.markdown("**Agent**")
    # Fixed temperature (tune here if you want)
    temp = 0.0
    # Hide debug output by default (set to True below if you need debugging)
    show_debug = False


with col2:
    st.title("Reservation Agent (Demo)")
    if "history" not in st.session_state:
        st.session_state.history = []

    # chat input area
    user_input = st.text_input("You:", key="user_input_text")
    if st.button("Send", key="send_btn"):
        if user_input.strip() == "":
            st.warning("Say something!")
        else:
            # append user message
            st.session_state.history.append(("user", user_input))

            # Build prompt context (last 10 turns)
            prompt_text = "\n".join([f"{u}: {m}" for (u,m) in st.session_state.history[-10:]])

            # -------- PRE-PARSER (improved) --------
            user_text = user_input.lower()
            known_cuisines = ["indian","italian","chinese","mexican","thai","japanese","seafood","continental","mediterranean"]
            known_areas = [a for a in ["Andheri","Bandra","Lower Parel","Connaught Place","Koramangala","Jayanagar","MG Road","Indiranagar","Noida Sector 18","Colaba"]]
            pre_args = {}
            # cuisine
            for c in known_cuisines:
                if c in user_text:
                    pre_args["cuisine"] = c
                    break
            # location/area
            for a in known_areas:
                if a.lower() in user_text:
                    pre_args["location"] = a
                    break
            # party size
            m = re.search(r'(\d+)\s*(?:people|person|guests|pax)', user_text)
            if m:
                pre_args["party_size"] = int(m.group(1))
            # parse date/time robustly
            dt = parse_date_time(user_text)
            if "date" in dt:
                pre_args["date"] = dt["date"]
            if "time" in dt:
                pre_args["time"] = dt["time"]

            # If user says "tonight" but no explicit time, set a default 20:00
            if "tonight" in user_text and "time" not in pre_args:
                pre_args["time"] = "20:00"
                pre_args["date"] = pre_args.get("date", datetime.now().date().isoformat())

            # Call LLM with the prompt (pass pre-args by including them into the prompt to bias output)
            # We add a short "CONTEXT" hint so the model is more likely to use the inferred fields.
            context_hint = f"CONTEXT: inferred_args={json.dumps(pre_args)}"
            llm_input = context_hint + "\n" + prompt_text
            llm_response = call_llm_openrouter(llm_input, temperature=float(temp), max_tokens=512)

            # Debug in sidebar
            if show_debug:
                st.sidebar.header("Debug")
                st.sidebar.write("User raw text:", user_input)
                st.sidebar.write("Pre-parsed args:", pre_args)
                st.sidebar.write("LLM raw response:", llm_response)

            parsed = safe_parse_json(llm_response)

            if parsed is None:
                # no tool call — treat as conversational reply
                st.session_state.history.append(("agent", llm_response))
            else:
                if show_debug:
                    st.sidebar.write("Parsed tool call:", parsed)
                parsed_args = parsed.get("args", {}) if isinstance(parsed, dict) else {}
                # Merge: prefer user's explicit words (pre_args) only if the LLM did not provide that field
                for k, v in pre_args.items():
                    if k not in parsed_args or parsed_args.get(k) in ("", None, 0):
                        parsed_args[k] = v
                parsed["args"] = parsed_args

                # Execute tool
                tool = parsed.get("tool")
                args = parsed.get("args", {})
                st.session_state.history.append(("agent","(executing tool)"))

                tool_output = None
                if tool == "recommend":
                    tool_output = tool_recommend(args)
                elif tool == "check_availability":
                    if not all(k in args for k in ("restaurant_id","date","time","party_size")):
                        st.session_state.history.append(("agent","Missing args for availability check. Please include restaurant id, date, time, and party size."))
                    else:
                        tool_output = tool_check_availability(args)
                elif tool == "create_reservation":
                    if not all(k in args for k in ("restaurant_id","date","time","party_size","name")):
                        st.session_state.history.append(("agent","Missing args for reservation. Please include restaurant_id, date, time, party_size and name."))
                    else:
                        tool_output = tool_create_reservation(args)
                elif tool == "show_bookings":
                    tool_output = tool_show_bookings(args)
                else:
                    st.session_state.history.append(("agent","Unknown tool requested by LLM."))

                # If we have a tool_output, ask the model to compose a final reply using TOOL_RESULT
                if tool_output is not None:
                    # Provide transparency about inferred fields so user sees what was assumed
                    assumed = {k:v for k,v in parsed_args.items() if k in ("date","time","party_size","location","cuisine") and k in pre_args}
                    result_msg = "TOOL_RESULT: " + json.dumps(tool_output)
                    followup_prompt = (prompt_text + "\n" + json.dumps(parsed) + "\n" + 
                                       "ASSUMED_FIELDS: " + json.dumps(assumed) + "\n" + result_msg + 
                                       "\nAssistant: Please provide a brief, friendly user-facing reply summarizing the result.")
                    final_llm_text = call_llm_openrouter(followup_prompt, temperature=0.0, max_tokens=256)
                    final_parsed = safe_parse_json(final_llm_text)
                    if final_parsed:
                        st.session_state.history.append(("agent", str(tool_output)))
                    else:
                        st.session_state.history.append(("agent", final_llm_text))

    # display chat history with simple bubbles
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for speaker, text in st.session_state.history[-50:]:
        if speaker == "user":
            st.markdown(f'<div class="meta">You</div><div class="user">{st.session_state.get("user_display_prefix","")}{text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="meta">Agent</div><div class="agent">{text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

