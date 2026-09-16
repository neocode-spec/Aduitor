
import os
import sqlite3
from datetime import datetime, date
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from google import genai


# ============================================================
# ADIUTOR
# AI ASSISTANT FOR WHATSAPP
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

FREE_DAILY_MESSAGES = int(
    os.getenv("FREE_DAILY_MESSAGES", "20")
)

DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    "adiutor.db"
)

WHATSAPP_VERIFY_TOKEN = os.getenv(
    "WHATSAPP_VERIFY_TOKEN",
    "change_this_later"
)

WHATSAPP_ACCESS_TOKEN = os.getenv(
    "WHATSAPP_ACCESS_TOKEN",
    ""
)

WHATSAPP_PHONE_NUMBER_ID = os.getenv(
    "WHATSAPP_PHONE_NUMBER_ID",
    ""
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Adiutor",
    description="AI assistant designed to live inside WhatsApp.",
    version="0.1.0"
)


# ============================================================
# GEMINI
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_database():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            premium INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_database()


# ============================================================
# USER MANAGEMENT
# ============================================================

def get_or_create_user(user_id: str):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE user_id = ?",
        (user_id,)
    )

    user = cursor.fetchone()

    if user is None:

        cursor.execute(
            """
            INSERT INTO users
            (user_id, premium, created_at)
            VALUES (?, 0, ?)
            """,
            (
                user_id,
                datetime.utcnow().isoformat()
            )
        )

        conn.commit()

        cursor.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        )

        user = cursor.fetchone()

    conn.close()

    return user


# ============================================================
# DAILY USAGE
# ============================================================

def get_daily_usage(user_id: str) -> int:

    conn = get_db()
    cursor = conn.cursor()

    today = date.today().isoformat()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM messages
        WHERE user_id = ?
        AND role = 'user'
        AND DATE(created_at) = ?
        """,
        (
            user_id,
            today
        )
    )

    count = cursor.fetchone()[0]

    conn.close()

    return count


# ============================================================
# MESSAGE HISTORY
# ============================================================

def save_message(
    user_id: str,
    role: str,
    content: str
):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO messages
        (user_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            role,
            content,
            datetime.utcnow().isoformat()
        )
    )

    conn.commit()
    conn.close()


def get_history(
    user_id: str,
    limit: int = 20
):

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            user_id,
            limit
        )
    )

    rows = cursor.fetchall()

    conn.close()

    rows = list(reversed(rows))

    return rows


# ============================================================
# ADIUTOR SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are Adiutor, an AI personal assistant designed to live inside WhatsApp.

Your job is to help users through natural conversation.

You can help with:

- Writing emails
- Rewriting messages
- Checking grammar
- Continuing drafts
- Summarizing information
- Explaining difficult topics
- Brainstorming
- Research assistance
- Planning
- Organizing information
- Business assistance
- General questions
- Turning rough ideas into polished writing

Behavior:

1. Be useful and direct.
2. Understand conversational context.
3. If the user says "continue", use the previous conversation context.
4. If the user asks to rewrite something, provide the rewritten version directly.
5. Do not pretend you performed an external action when you did not.
6. Do not claim that an email was sent unless an actual email service confirms it.
7. Do not claim that you searched the web unless a web-search tool was actually used.
8. Keep WhatsApp responses readable and reasonably concise.
9. Use plain text and simple formatting that works well in WhatsApp.
10. If the user asks for a document, produce clean copy that can be copied easily.
11. Remember relevant context within the conversation.
12. Be honest about limitations.

You are Adiutor, not a generic chatbot.
"""


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):

    message: str

    user_id: str = "browser_user"


class ChatResponse(BaseModel):

    response: str

    usage: int

    daily_limit: Optional[int] = None

    premium: bool = False


# ============================================================
# GEMINI RESPONSE
# ============================================================

def generate_response(
    user_id: str,
    user_message: str
) -> str:

    if gemini_client is None:

        raise RuntimeError(
            "Gemini API key is not configured."
        )

    history = get_history(
        user_id,
        limit=20
    )

    conversation = []

    for item in history:

        role = item["role"]

        if role == "user":
            conversation.append(
                f"User: {item['content']}"
            )

        elif role == "assistant":
            conversation.append(
                f"Adiutor: {item['content']}"
            )

    conversation.append(
        f"User: {user_message}"
    )

    prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + "\n".join(conversation)
        + "\n\nAdiutor:"
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    user = get_or_create_user(
        request.user_id
    )

    premium = bool(user["premium"])

    usage = get_daily_usage(
        request.user_id
    )

    # --------------------------------------------------------
    # FREE USER LIMIT
    # --------------------------------------------------------

    if not premium and usage >= FREE_DAILY_MESSAGES:

        return ChatResponse(
            response=(
                "You have reached your free daily limit "
                f"of {FREE_DAILY_MESSAGES} messages.\n\n"
                "Upgrade your Adiutor plan to continue "
                "using the assistant."
            ),
            usage=usage,
            daily_limit=FREE_DAILY_MESSAGES,
            premium=False
        )

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    save_message(
        request.user_id,
        "user",
        message
    )

    # --------------------------------------------------------
    # GENERATE AI RESPONSE
    # --------------------------------------------------------

    try:

        answer = generate_response(
            request.user_id,
            message
        )

    except Exception as error:

        error_text = str(error).lower()

        # Gemini quota / rate-limit handling
        if (
            "429" in error_text
            or "quota" in error_text
            or "rate" in error_text
            or "resource exhausted" in error_text
        ):

            return ChatResponse(
                response=(
                    "Adiutor is temporarily at its AI usage "
                    "limit. Please try again later."
                ),
                usage=usage,
                daily_limit=(
                    None
                    if premium
                    else FREE_DAILY_MESSAGES
                ),
                premium=premium
            )

        print(
            "Gemini error:",
            error
        )

        return ChatResponse(
            response=(
                "Adiutor could not process that request "
                "right now. Please try again."
            ),
            usage=usage,
            daily_limit=(
                None
                if premium
                else FREE_DAILY_MESSAGES
            ),
            premium=premium
        )

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    save_message(
        request.user_id,
        "assistant",
        answer
    )

    new_usage = get_daily_usage(
        request.user_id
    )

    return ChatResponse(
        response=answer,
        usage=new_usage,
        daily_limit=(
            None
            if premium
            else FREE_DAILY_MESSAGES
        ),
        premium=premium
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "online",
        "service": "Adiutor",
        "gemini_configured": bool(
            GEMINI_API_KEY
        ),
        "whatsapp_configured": bool(
            WHATSAPP_ACCESS_TOKEN
            and WHATSAPP_PHONE_NUMBER_ID
        )
    }


# ============================================================
# ROOT PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>Adiutor</title>

        <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
        >

        <style>

            body {
                margin: 0;
                background: #050505;
                color: white;
                font-family: Arial, sans-serif;
            }

            .container {
                max-width: 800px;
                margin: auto;
                padding: 40px 20px;
            }

            h1 {
                margin-bottom: 5px;
            }

            .status {
                color: #888;
                margin-bottom: 30px;
            }

            textarea {
                width: 100%;
                height: 100px;
                background: #111;
                color: white;
                border: 1px solid #333;
                border-radius: 10px;
                padding: 15px;
                box-sizing: border-box;
                resize: vertical;
            }

            button {
                margin-top: 10px;
                padding: 12px 20px;
                background: white;
                color: black;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
            }

            .response {
                margin-top: 25px;
                background: #111;
                border: 1px solid #222;
                border-radius: 10px;
                padding: 20px;
                white-space: pre-wrap;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1>Adiutor</h1>

            <div class="status">
                AI assistant backend
            </div>

            <textarea
                id="message"
                placeholder="Talk to Adiutor..."
            ></textarea>

            <br>

            <button onclick="sendMessage()">
                Send
            </button>

            <div
                id="response"
                class="response"
            >
                Waiting for a message...
            </div>

        </div>


        <script>

            async function sendMessage() {

                const message =
                    document.getElementById(
                        "message"
                    ).value.trim();

                if (!message) {
                    return;
                }

                const responseBox =
                    document.getElementById(
                        "response"
                    );

                responseBox.innerText =
                    "Adiutor is thinking...";

                try {

                    const response =
                        await fetch(
                            "/api/chat",
                            {
                                method: "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body: JSON.stringify({
                                    message: message,
                                    user_id: "browser_user"
                                })
                            }
                        );

                    const data =
                        await response.json();

                    if (!response.ok) {

                        responseBox.innerText =
                            data.detail ||
                            "Something went wrong.";

                        return;
                    }

                    responseBox.innerText =
                        data.response;

                } catch (error) {

                    responseBox.innerText =
                        "Could not connect to Adiutor.";

                }

            }

        </script>

    </body>

    </html>
    """


# ============================================================
# WHATSAPP WEBHOOK VERIFICATION
# ============================================================

@app.get("/webhook")
async def verify_whatsapp_webhook(
    request: Request
):

    params = request.query_params

    mode = params.get(
        "hub.mode"
    )

    token = params.get(
        "hub.verify_token"
    )

    challenge = params.get(
        "hub.challenge"
    )

    if (
        mode == "subscribe"
        and token == WHATSAPP_VERIFY_TOKEN
    ):

        return PlainTextResponse(
            challenge or ""
        )

    raise HTTPException(
        status_code=403,
        detail="Webhook verification failed."
    )


# ============================================================
# WHATSAPP WEBHOOK
# ============================================================

@app.post("/webhook")
async def whatsapp_webhook(
    request: Request
):

    data = await request.json()

    print(
        "WhatsApp webhook received:"
    )

    print(data)

    # --------------------------------------------------------
    # WHATSAPP MESSAGE PROCESSING WILL BE ADDED HERE.
    #
    # This endpoint currently receives WhatsApp events,
    # but it does not yet send replies back through Meta.
    #
    # The next WhatsApp integration stage will:
    #
    # 1. Read the incoming WhatsApp message
    # 2. Identify the sender
    # 3. Send the message to Adiutor
    # 4. Generate the Gemini response
    # 5. Send the response back through WhatsApp
    # --------------------------------------------------------

    return {
        "status": "received"
    }


# ============================================================
# STARTUP MESSAGE
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        ),
        reload=False
    )
