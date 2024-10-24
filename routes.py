# routes.py
from fastapi import APIRouter, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json
from helper import find_ticker, fetch_fundamentals, scrape_news, initialize_chat_model

app = FastAPI()

# Allow all origins for development purposes. Change these settings for production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, use specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

router = APIRouter()

chat_session = initialize_chat_model()

@router.get("/")
async def root():
    return {"message": "Hello World"}

@router.get("/lstm/{stock}")
async def lstm(stock: str, period: int, interval: int):
    ticker = find_ticker(stock)
    return {
        "ticker": ticker,
        "period": period,
        "interval": interval
    }

@router.get("/fundamentals/{stock}")
async def fundamentals(stock: str):
    ticker = find_ticker(stock)
    return fetch_fundamentals(ticker)

@router.get("/summarize/{stock}")
async def summarize(stock: str):
    with open('config.json') as config_file:
        config = json.load(config_file)
        
    gemini_prompt = config["GEMINI_PROMPT"]
    
    print("================================")
    print("Stock: ", stock)
    print("================================")
    print("scraping news...")

    news = scrape_news(stock)

    print("news scraped", news)

    ticker = find_ticker(stock)

    print("finding fundamentals...")
    fundamentals = fetch_fundamentals(ticker)

    print("fundamentals found", fundamentals)

    response = ""

    if chat_session:
        output = chat_session.send_message(str(gemini_prompt) + str(news["headlines"]) + json.dumps(fundamentals))
        response = output.text
        
    return {
        "summary": response,
        "fundamentals": fundamentals
    }

# Include router
app.include_router(router)
