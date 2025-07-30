from fastapi import FastAPI, File, UploadFile, HTTPException, Header
import speech_recognition as sr
from pydub import AudioSegment
import io
import os
import re
import json
import dateparser
import openai
import datetime
import requests
import pytz
import uvicorn
from math import sqrt

# ENV Secrets
openai.api_key = os.getenv("OPENAI_API_KEY")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "defaultpass")
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN")

# App setup
app = FastAPI()

# Timezone setup
EST = pytz.timezone("America/New_York")

# Urls
TOKEN_URL = "https://oauth2.googleapis.com/token"
EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"

COLOR_HEX_MAP = {
    "1": "#a4bdfc", "2": "#7ae7bf", "3": "#dbadff", "4": "#ff887c",
    "5": "#fbd75b", "6": "#ffb878", "7": "#46d6db", "8": "#e1e1e1",
    "9": "#5484ed", "10": "#51b749", "11": "#dc2127"
}

CSS_COLOR_NAMES = {
    "red": "#ff0000",
    "lightred": "#ff6666",
    "darkred": "#8b0000",

    "blue": "#0000ff",
    "lightblue": "#add8e6",
    "darkblue": "#00008b",

    "green": "#008000",
    "lightgreen": "#90ee90",
    "darkgreen": "#006400",

    "yellow": "#ffff00",
    "lightyellow": "#ffffe0",
    "darkyellow": "#bdb76b",

    "orange": "#ffa500",
    "lightorange": "#ffb347",
    "darkorange": "#ff8c00",

    "purple": "#800080",
    "lightpurple": "#dab5d3",
    "darkpurple": "#4b0082",

    "pink": "#ffc0cb",
    "lightpink": "#ffb6c1",
    "darkpink": "#ff69b4",

    "gray": "#808080",
    "lightgray": "#d3d3d3",
    "darkgray": "#a9a9a9",

    "black": "#000000",
    "white": "#ffffff",
    "teal": "#008080"
}

# Utility functions

def get_nearest_color_id(hex_input):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    target_rgb = hex_to_rgb(hex_input)
    min_dist, best_id = float("inf"), "1"

    for cid, hex_val in COLOR_HEX_MAP.items():
        r, g, b = hex_to_rgb(hex_val)
        dist = sqrt((target_rgb[0] - r) ** 2 + (target_rgb[1] - g) ** 2 + (target_rgb[2] - b) ** 2)
        if dist < min_dist:
            min_dist = dist
            best_id = cid

    return best_id

def get_access_token():
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token"
    }
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json().get("access_token")

def add_event(access_token, summary, color_hex, start_datetime_est):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    localized = EST.localize(start_datetime_est)
    iso_dt = localized.isoformat()
    color_id = get_nearest_color_id(color_hex or "#008080")

    event = {
        "summary": summary,
        "start": {"dateTime": iso_dt, "timeZone": "America/New_York"},
        "end": {"dateTime": iso_dt, "timeZone": "America/New_York"},
        "colorId": color_id
    }
    response = requests.post(EVENTS_URL, headers=headers, json=event)
    response.raise_for_status()
    return response.json()

def call_gpt(user_input):
    client = openai.OpenAI()
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = (
        "Assume today's date is " + now + ".\n"
        "Extract from this text the title, the color (as hex if named or hex given, else null), "
        "and the datetime (ISO format, or null if none).\n\n"
        f"Text: \"{user_input}\"\n\n"
        "Return ONLY a JSON object with keys: title, color, datetime."
        "If the date is not specified, set it to today at noon or tomorrow at noon if it's already past noon."
        "If the color is not mentioned, set it to teal."
        "If datetime is before or at now, return null for datetime."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    content = response.choices[0].message.content

    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:-1])
        
    print("GPT response:", content)
    return json.loads(response.choices[0].message.content)

# FastAPI routes

@app.post("/speech-to-calendar")
async def process_audio(
    file: UploadFile = File(...), 
    authorization: str = Header(..., alias="Authorization")
):
    if authorization != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/flac", "audio/mpeg", "audio/mp3"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    audio_bytes = await file.read()

    if file.content_type not in ["audio/wav", "audio/x-wav"]:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        audio_data = sr.AudioFile(wav_io)
    else:
        audio_data = sr.AudioFile(io.BytesIO(audio_bytes))

    recognizer = sr.Recognizer()
    with audio_data as source:
        audio_record = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_record)
    except sr.UnknownValueError:
        return {"text": "", "error": "Could not understand audio"}
    except sr.RequestError:
        return {"text": "", "error": "Speech Recognition service error"}

    gpt_data = call_gpt(text)
    dt = gpt_data.get("datetime")
    if not dt:
        return {"text": text, "error": "Invalid or past datetime in input"}

    access_token = get_access_token()
    summary = gpt_data.get("title") or "No Title"
    color_hex = gpt_data.get("color") or "#008080"
    start_dt = datetime.datetime.fromisoformat(dt)
    event_result = add_event(access_token, summary, color_hex, start_dt)

    return {"text": text, "parsed": gpt_data, "calendar_event": event_result}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)