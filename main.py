
import os
import sqlite3
from datetime import datetime, date, timezone
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


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
).strip()

try:
    FREE_DAILY_MESSAGES = int(
        os.getenv(
            "FREE_DAILY_MESSAGES",
            "20"
        )
    )
except ValueError:
    FREE_DAILY_MESSAGES = 20


DATABASE_PATH = os.getenv(
    "DATABASE_PATH",
    "adiutor.db"
).strip()


WHATSAPP_VERIFY_TOKEN = os.getenv(
    "WHATSAPP_VERIFY_TOKEN",
    "change_this_later"
).strip()


WHATSAPP_ACCESS_TOKEN = os.getenv(
    "WHATSAPP_ACCESS_TOKEN",
    ""
).strip()


WHATSAPP_PHONE_NUMBER_ID = os.getenv(
    "WHATSAPP_PHONE_NUMBER_ID",
    ""
).strip()


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

gemini_client: Optional[genai.Client] = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print("Gemini client initialized.")

    except Exception as error:
        print(
            "Gemini initialization failed:",
            error
        )

else:
    print(
        "WARNING: GEMINI_API_KEY is not configured."
    )


# ============================================================
# DATABASE
# ============================================================

def get_db():
    """
    Open a connection to the Adiutor SQLite database.
    """

    conn = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_database():
    """
    Create Adiutor database tables if they do not exist.
    """

    conn = get_db()

    try:
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

    finally:
        conn.close()


init_database()


# ============================================================
# TIME
# ============================================================

def utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO 8601 string.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# USER MANAGEMENT
# ============================================================

def get_or_create_user(user_id: str):

    conn = get_db()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM users
            WHERE user_id = ?
            """,
            (user_id,)
        )

        user = cursor.fetchone()

        if user is None:

            cursor.execute(
                """
                INSERT INTO users
                (
                    user_id,
                    premium,
                    created_at
                )
                VALUES (?, 0, ?)
                """,
                (
                    user_id,
                    utc_now_iso()
                )
            )

            conn.commit()

            cursor.execute(
                """
                SELECT *
                FROM users
                WHERE user_id = ?
                """,
                (user_id,)
            )

            user = cursor.fetchone()

        return user

    finally:
        conn.close()


# ============================================================
# DAILY USAGE
# ============================================================

def get_daily_usage(user_id: str) -> int:

    conn = get_db()

    try:
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

        return count

    finally:
        conn.close()


# ============================================================
# MESSAGE STORAGE
# ============================================================

def save_message(
    user_id: str,
    role: str,
    content: str
):
    """
    Save a user or assistant message.
    """

    conn = get_db()

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO messages
            (
                user_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                role,
                content,
                utc_now_iso()
            )
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================
# MESSAGE HISTORY
# ============================================================

def get_history(
    user_id: str,
    limit: int = 20
):

    conn = get_db()

    try:
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

    finally:
        conn.close()

    return list(reversed(rows))


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
11. Remember relevant context within the available conversation history.
12. Be honest about limitations.
13. Do not invent external information or actions.
14. You are Adiutor, not a generic chatbot.
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

    user_id = request.user_id.strip()

    if not user_id:

        raise HTTPException(
            status_code=400,
            detail="User ID cannot be empty."
        )

    # --------------------------------------------------------
    # GET OR CREATE USER
    # --------------------------------------------------------

    user = get_or_create_user(
        user_id
    )

    premium = bool(
        user["premium"]
    )

    # --------------------------------------------------------
    # CHECK DAILY USAGE
    # --------------------------------------------------------

    usage = get_daily_usage(
        user_id
    )

    if (
        not premium
        and usage >= FREE_DAILY_MESSAGES
    ):

        return ChatResponse(
            response=(
                "You have reached your free daily "
                f"limit of {FREE_DAILY_MESSAGES} "
                "messages today.\n\n"
                "Upgrade your Adiutor plan to "
                "continue using the assistant."
            ),
            usage=usage,
            daily_limit=FREE_DAILY_MESSAGES,
            premium=False
        )

    # --------------------------------------------------------
    # GENERATE AI RESPONSE
    #
    # IMPORTANT:
    # We generate the response BEFORE saving the user
    # message. This prevents failed Gemini requests from
    # consuming the user's daily message allowance.
    # --------------------------------------------------------

    try:

        answer = generate_response(
            user_id,
            message
        )

    except Exception as error:

        error_text = str(error).lower()

        # ----------------------------------------------------
        # GEMINI QUOTA / RATE LIMIT
        # ----------------------------------------------------

        if (
            "429" in error_text
            or "quota" in error_text
            or "rate" in error_text
            or "resource exhausted" in error_text
            or "too many requests" in error_text
        ):

            return ChatResponse(
                response=(
                    "Adiutor is temporarily unable "
                    "to use its AI service because "
                    "the current AI usage limit has "
                    "been reached.\n\n"
                    "Please try again later."
                ),
                usage=usage,
                daily_limit=(
                    None
                    if premium
                    else FREE_DAILY_MESSAGES
                ),
                premium=premium
            )

        # ----------------------------------------------------
        # API KEY / AUTHENTICATION ERROR
        # ----------------------------------------------------

        if (
            "api key" in error_text
            or "unauthorized" in error_text
            or "authentication" in error_text
            or "permission denied" in error_text
        ):

            print(
                "Gemini authentication error:",
                error
            )

            return ChatResponse(
                response=(
                    "Adiutor's AI service is not "
                    "properly configured right now."
                ),
                usage=usage,
                daily_limit=(
                    None
                    if premium
                    else FREE_DAILY_MESSAGES
                ),
                premium=premium
            )

        # ----------------------------------------------------
        # GENERAL GEMINI ERROR
        # ----------------------------------------------------

        print(
            "Gemini error:",
            error
        )

        return ChatResponse(
            response=(
                "Adiutor could not process that "
                "request right now. Please try again."
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
    # SAVE SUCCESSFUL CONVERSATION
    # --------------------------------------------------------

    save_message(
        user_id,
        "user",
        message
    )

    save_message(
        user_id,
        "assistant",
        answer
    )

    # --------------------------------------------------------
    # UPDATED USAGE
    # --------------------------------------------------------

    new_usage = get_daily_usage(
        user_id
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
        "gemini_model": GEMINI_MODEL,
        "whatsapp_configured": bool(
            WHATSAPP_ACCESS_TOKEN
            and WHATSAPP_PHONE_NUMBER_ID
        )
    }


# ============================================================
# ROOT PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
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

            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }

            .response {
                margin-top: 25px;
                background: #111;
                border: 1px solid #222;
                border-radius: 10px;
                padding: 20px;
                white-space: pre-wrap;
                min-height: 40px;
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

            <button
                id="sendButton"
                onclick="sendMessage()"
            >
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

                const responseBox =
                    document.getElementById(
                        "response"
                    );

                const sendButton =
                    document.getElementById(
                        "sendButton"
                    );

                if (!message) {
                    return;
                }

                sendButton.disabled = true;

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

                    document.getElementById(
                        "message"
                    ).value = "";

                } catch (error) {

                    responseBox.innerText =
                        "Could not connect to Adiutor.";

                } finally {

                    sendButton.disabled = false;

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

    try:

        data = await request.json()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload."
        )

    print(
        "WhatsApp webhook received:"
    )

    print(data)

    # --------------------------------------------------------
    # WHATSAPP MESSAGE PROCESSING
    #
    # This currently receives Meta webhook events.
    #
    # The next WhatsApp integration stage will:
    #
    # 1. Read the incoming WhatsApp message
    # 2. Identify the sender
    # 3. Pass the message to Adiutor
    # 4. Generate the Gemini response
    # 5. Send the response back through Meta
    # --------------------------------------------------------

    return {
        "status": "received"
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 50)
    print("ADIUTOR STARTING")
    print("=" * 50)

    print(
        "Gemini configured:",
        bool(GEMINI_API_KEY)
    )

    print(
        "Gemini model:",
        GEMINI_MODEL
    )

    print(
        "Free daily messages:",
        FREE_DAILY_MESSAGES
    )

    print(
        "Database:",
        DATABASE_PATH
    )

    print(
        "WhatsApp configured:",
        bool(
            WHATSAPP_ACCESS_TOKEN
            and WHATSAPP_PHONE_NUMBER_ID
        )
    )

    print("=" * 50)


# ============================================================
# DIRECT START
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

