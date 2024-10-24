import requests
from bs4 import BeautifulSoup
from typing import Dict
import requests
import json
import google.generativeai as genai

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

def initialize_chat_model(config_path='config.json'):
    """
    Initializes the generative AI model and starts a chat session.
    
    Args:
        config_path (str): Path to the JSON configuration file.
    
    Returns:
        chat_session: An instance of the chat session.
    """
    try:
        # Load the API key from the config file
        with open(config_path) as config_file:
            config = json.load(config_file)
        
        api_key = config["GEMINI_API_KEY"]

        # Configure the API with the loaded key
        genai.configure(api_key=api_key)

        # Create the model configuration
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        # Create the generative model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
        )

        # Start a chat session
        chat_session = model.start_chat(history=[])

        return chat_session

    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading API key: {e}")
        return None

def find_ticker(s):

    url = "http://query2.finance.yahoo.com/v1/finance/search?q=" + s
    # Send a GET request to the Yahoo Finance search API
    response = requests.get(url, headers=headers)
    data = ""    
    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response to JSON
        
        try:
            data = response.json()['quotes'][0]["symbol"]
        except:
            pass
        
    return data


def fetch_fundamentals(ticker: str) -> Dict:

    url = f'https://finance.yahoo.com/quote/{ticker}'

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Initialize the result dictionary with organized categories
        result = {
            "valuation_metrics": {
                "market_cap": None,
                "enterprise_value": None,
                "trailing_pe": None,
                "forward_pe": None,
                "peg_ratio": None,
                "price_to_sales": None,
                "price_to_book": None,
                "enterprise_value_to_revenue": None,
                "enterprise_value_to_ebitda": None
            },
            "trading_info": {
                "days_range": None,
                "fifty_two_week_range": None,
                "volume": None,
                "avg_volume": None
            },
            "financial_metrics": {
                "profit_margin": None,
                "return_on_assets": None,
                "return_on_equity": None,
                "revenue": None,
                "net_income": None,
                "diluted_eps": None,
                "total_cash": None,
                "debt_to_equity": None,
                "levered_free_cash_flow": None
            }
        }

        # Helper function to clean values
        def clean_value(value: str) -> str:
            return value.strip().replace('(', '').replace(')', '')

        # Process statistics section
        statistics_section = soup.find('div', {'data-testid': 'quote-statistics'})
        if statistics_section:
            stats_text = [text.strip() for text in statistics_section.stripped_strings if text.strip()]
            
            # Map statistics data
            mapping = {
                "Market Cap (intraday)": ("valuation_metrics", "market_cap"),
                "PE Ratio (TTM)": ("valuation_metrics", "trailing_pe"),
                "Day's Range": ("trading_info", "days_range"),
                "52 Week Range": ("trading_info", "fifty_two_week_range"),
                "Volume": ("trading_info", "volume"),
                "Avg. Volume": ("trading_info", "avg_volume"),
                "EPS (TTM)": ("financial_metrics", "diluted_eps")
            }
            
            for i in range(0, len(stats_text)-1, 2):
                key = stats_text[i]
                value = clean_value(stats_text[i+1])
                
                if key in mapping:
                    category, field = mapping[key]
                    result[category][field] = value

        # Process panel items
        panel_items = soup.select('.panel li')
        panel_mapping = {
            "Market Cap": ("valuation_metrics", "market_cap"),
            "Enterprise Value": ("valuation_metrics", "enterprise_value"),
            "Trailing P/E": ("valuation_metrics", "trailing_pe"),
            "Forward P/E": ("valuation_metrics", "forward_pe"),
            "PEG Ratio": ("valuation_metrics", "peg_ratio"),
            "Price/Sales": ("valuation_metrics", "price_to_sales"),
            "Price/Book": ("valuation_metrics", "price_to_book"),
            "Enterprise Value/Revenue": ("valuation_metrics", "enterprise_value_to_revenue"),
            "Enterprise Value/EBITDA": ("valuation_metrics", "enterprise_value_to_ebitda"),
            "Profit Margin": ("financial_metrics", "profit_margin"),
            "Return on Assets": ("financial_metrics", "return_on_assets"),
            "Return on Equity": ("financial_metrics", "return_on_equity"),
            "Revenue": ("financial_metrics", "revenue"),
            "Net Income Avi to Common": ("financial_metrics", "net_income"),
            "Diluted EPS": ("financial_metrics", "diluted_eps"),
            "Total Cash": ("financial_metrics", "total_cash"),
            "Total Debt/Equity": ("financial_metrics", "debt_to_equity"),
            "Levered Free Cash Flow": ("financial_metrics", "levered_free_cash_flow")
        }

        for item in panel_items:
            text = item.get_text().strip()
            key, value = [part.strip() for part in text.split('  ') if part.strip()]
            
            # Clean the key by removing parenthetical information
            clean_key = key.split('(')[0].strip()
            
            if clean_key in panel_mapping:
                category, field = panel_mapping[clean_key]
                result[category][field] = clean_value(value)

        return result

    except requests.RequestException as e:
        return {"error": f"Failed to retrieve data: {str(e)}"}
   



def scrape_news(ticker):
    url = 'https://news.google.com/search?q=' + ticker
    
    # Make the request to Yahoo Finance
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract headlines from news section
        news_items = soup.select('[data-n-tid="29"]')  # Adjust the selector as needed
        
        # Combine inner text of all selected news items
        headlines = [item.get_text(strip=True) for item in news_items]
        
        # Return the headlines as JSON
        return {"headlines": headlines}
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None
    


