# app.py
"""
GoodFoods Reservation Agent
Streamlit app with OpenAI tool-calling integration (recommend / availability / reservation)
Requirements: streamlit, pandas, requests, openai
Set environment variable OPENAI_API_KEY to your OpenAI API key before running.
"""

import os
import streamlit as st
import sqlite3
import json
import re
from datetime import datetime
import pandas as pd
import requests
import openai
import time
from openai import OpenAI 
# ---------- Configuration ----------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLAMA_MODEL = "x-ai/grok-4.1-fast"

DB = "goodfoods.db"

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

# call this on startup
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


# ---------- LLM integration (OpenAI) ----------
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

# cache discovered model name to avoid repeated model-list calls
_cached_model_name = None
_last_model_fetch = 0

def discover_model(preferred=LLAMA_MODEL):
    global _cached_model_name, _last_model_fetch
    # return cached if recent
    if _cached_model_name and (time.time() - _last_model_fetch) < 300:
        return _cached_model_name

    # first, try preferred model by probing with a minimal completion call
    probe_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    probe_payload = {
        "model": preferred,
        "messages": [{"role":"system","content":"probe"},{"role":"user","content":"hi"}],
        "max_tokens": 1
    }
    try:
        r = requests.post(probe_url, headers=headers, json=probe_payload, timeout=15)
        # if status 200, preferred works
        if r.status_code == 200:
            _cached_model_name = preferred
            _last_model_fetch = time.time()
            print("Model probe success for preferred model:", preferred)
            return _cached_model_name
        else:
            print("Preferred model probe status:", r.status_code, r.text[:400])
    except Exception as e:
        print("Preferred model probe exception:", e)

    # If preferred failed, fetch list of models and pick a reasonable fallback
    models_resp = fetch_openrouter_models()
    _last_model_fetch = time.time()
    if not models_resp:
        return preferred  # we couldn't fetch list; return preferred to try again later

    # models_resp often contains a list under 'models' or is itself a list
    model_entries = None
    if isinstance(models_resp, dict) and "models" in models_resp:
        model_entries = models_resp["models"]
    elif isinstance(models_resp, list):
        model_entries = models_resp
    else:
        # sometimes it's nested; try to find list
        for v in models_resp.values():
            if isinstance(v, list):
                model_entries = v
                break
    if not model_entries:
        return preferred

    # search for names containing 'llama' or '3.3' or 'instruct' or 'llama-3'
    candidates = []
    for m in model_entries:
        # model entry might be dict with 'id' or 'name'
        name = None
        if isinstance(m, dict):
            name = m.get("id") or m.get("name") or m.get("model")
        elif isinstance(m, str):
            name = m
        if not name:
            continue
        low = name.lower()
        if "llama" in low or "3.3" in low or "instruct" in low or "llama-3" in low:
            candidates.append(name)

    # if none found, fall back to first few models returned
    if not candidates:
        # try top-level model ids
        fallback = []
        for m in model_entries[:10]:
            if isinstance(m, dict):
                n = m.get("id") or m.get("name") or m.get("model")
                if n:
                    fallback.append(n)
            elif isinstance(m, str):
                fallback.append(m)
        candidates = fallback

    if candidates:
        _cached_model_name = candidates[0]
        print("Discovered fallback model:", _cached_model_name)
        return _cached_model_name

    return preferred

def call_llm_openrouter(conversation_text, temperature=0.0, max_tokens=512):
    """
    Robust OpenRouter caller: discovers a working model name if needed,
    and returns the assistant text or an ERROR_CALLING_LLM message.
    """
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
        err = f"ERROR_CALLING_LLM: network error: {e}"
        print(err)
        return err

    status = r.status_code
    raw = r.text
    try:
        data = r.json()
    except Exception:
        err = f"ERROR_CALLING_LLM: non-json response status={status} body={raw[:1000]}"
        print(err)
        try:
            st.sidebar.write("LLM raw response (non-json):", raw)
        except Exception:
            pass
        return err

    # Show raw JSON in sidebar for debugging
    try:
        st.sidebar.write("OpenRouter JSON:", data)
        st.sidebar.write("Using model:", model_to_use)
    except Exception:
        pass

    if status != 200:
        err_detail = data.get("error") or data
        err = f"ERROR_CALLING_LLM: status={status} detail={json.dumps(err_detail)[:1000]}"
        print(err)
        return err

    # extract assistant text (try typical shapes)
    if isinstance(data, dict) and "choices" in data and len(data["choices"])>0:
        choice = data["choices"][0]
        if isinstance(choice, dict) and "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
            return choice["message"]["content"]
        if isinstance(choice, dict) and "text" in choice:
            return choice["text"]

    # try other shapes
    if isinstance(data, dict) and "output" in data and isinstance(data["output"], list) and data["output"]:
        first = data["output"][0]
        if isinstance(first, dict) and "content" in first and isinstance(first["content"], list):
            for frag in first["content"]:
                if isinstance(frag, dict) and "text" in frag:
                    return frag["text"]

    # fallback: find first string in JSON
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

# ---------- Streamlit UI ----------
st.set_page_config(page_title="GoodFoods Reservation Agent", layout="wide")
st.title("GoodFoods — Reservation Agent (Demo)")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Restaurants")
    df = pd.read_sql("SELECT id,name,cuisine,capacity,area,price_bucket,rating FROM restaurants LIMIT 50", conn)
    st.dataframe(df)
    st.markdown("**Quick actions**")
    lookup_phone = st.text_input("Enter phone for lookup (left)", value="", key="lookup_phone")
    if st.button("Show my bookings (demo)", key="show_bookings_btn"):
        if lookup_phone.strip():
            out = tool_show_bookings({"phone":lookup_phone.strip()})
            st.write(out)
        else:
            st.warning("Enter a phone number to show bookings.")

with col2:
    st.subheader("Chat with Agent")
    if "history" not in st.session_state:
        st.session_state.history = []
    user_input = st.text_input("You:", key="user_input_text")

    # Send (unique key)
    if st.button("Send", key="send_btn"):
        if user_input.strip() == "":
            st.warning("Say something!")
        else:
            st.session_state.history.append(("user", user_input))

            # Build prompt context (last 10 turns)
            prompt_text = "\n".join([f"{u}: {m}" for (u,m) in st.session_state.history[-10:]])

            # PRE-PARSER (strong signals from raw user text)
            user_text = user_input.lower()
            known_cuisines = ["indian","italian","chinese","mexican","thai","japanese","seafood","continental","mediterranean"]
            known_areas = [a for a in ["Andheri","Bandra","Lower Parel","Connaught Place","Koramangala","Jayanagar","MG Road","Indiranagar","Noida Sector 18","Colaba"]]
            pre_args = {}
            for c in known_cuisines:
                if c in user_text:
                    pre_args["cuisine"] = c
                    break
            for a in known_areas:
                if a.lower() in user_text:
                    pre_args["location"] = a
                    break
            m = re.search(r'(\d+)\s*(?:people|person|guests|pax)', user_text)
            if m:
                pre_args["party_size"] = int(m.group(1))
            m2 = re.search(r'(\d{4}-\d{2}-\d{2})', user_text)
            if m2:
                pre_args["date"] = m2.group(1)
            m3 = re.search(r'(\d{1,2}[:.]\d{2})', user_text)
            if m3:
                pre_args["time"] = m3.group(1).replace('.', ':')

            # Call OpenAI
            llm_response = call_llm_openrouter(prompt_text)


            # DEBUG SIDEBAR
            st.sidebar.header("Debug")
            st.sidebar.write("User raw text:", user_input)
            st.sidebar.write("Pre-parsed args:", pre_args)
            st.sidebar.write("LLM raw response:", llm_response)

            parsed = safe_parse_json(llm_response)

            if parsed is None:
                # no tool call — treat as conversation reply
                st.session_state.history.append(("agent", llm_response))
            else:
                st.sidebar.write("Parsed tool call:", parsed)
                parsed_args = parsed.get("args", {}) if isinstance(parsed, dict) else {}

                # Merge (prefer user's explicit words)
                for k, v in pre_args.items():
                    parsed_args[k] = v

                parsed["args"] = parsed_args

                # Execute tool
                tool = parsed.get("tool")
                args = parsed.get("args", {})
                st.session_state.history.append(("agent","(executing tool)"))

                # Validate required fields for each tool before calling
                tool_output = None
                if tool == "recommend":
                    tool_output = tool_recommend(args)
                elif tool == "check_availability":
                    # basic validation
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
                    # Add the tool result to the prompt and ask model to create a human-friendly message
                    result_msg = "TOOL_RESULT: " + json.dumps(tool_output)
                    followup_prompt = prompt_text + "\n" + json.dumps(parsed) + "\n" + result_msg + "\nAssistant: Please provide a brief, friendly user-facing reply summarizing the result."
                    final_llm_text = call_llm_openrouter(followup_prompt, temperature=0.0, max_tokens=256)
                    # If model returns JSON again (should not), try parse; else treat as human-facing text
                    final_parsed = safe_parse_json(final_llm_text)
                    if final_parsed:
                        # if model incorrectly returned JSON, fallback to simple formatting
                        st.session_state.history.append(("agent", str(tool_output)))
                    else:
                        st.session_state.history.append(("agent", final_llm_text))

    # display conversation
    for speaker, text in st.session_state.history[-30:]:
        if speaker == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Agent:** {text}")
