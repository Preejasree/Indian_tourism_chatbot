from flask import Flask, request, jsonify, render_template,session
from typing import TypedDict
import os, requests, re
from dotenv import load_dotenv
import pandas as pd

from langgraph.graph import StateGraph, START, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings

from rapidfuzz import process, fuzz
from tavily import TavilyClient
from word2number import w2n
#from langchain_community.tools import DuckDuckGoSearchRun

#from ddgs import DDGS

# --------------------------------------------------
# FLASK APP
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = "travel_bot_secret"

# --------------------------------------------------
# ENV + MODEL
# --------------------------------------------------
load_dotenv()
HF_token = os.getenv("HF_TOKEN")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=HF_token,
        temperature=0.2,
        max_new_tokens=900
    )
)

#search_tool = DuckDuckGoSearchRun()

# --------------------------------------------------
# LOAD CSV (VALIDATION + RAG)
# --------------------------------------------------
CSV_PATH = "TopIndianPlacestoVisit.csv"
df = pd.read_csv(CSV_PATH)

CITY_COL = "City"
STATE_COL = "State"

cities = df[CITY_COL].dropna().str.lower().str.strip().tolist()
states = df[STATE_COL].dropna().str.lower().str.strip().tolist()

VALID_LOCATIONS = list(set(cities + states))

def load_csv_as_text(df):
    rows = []
    for _, row in df.iterrows():
        text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        rows.append(text)
    return rows

csv_texts = load_csv_as_text(df)

# --------------------------------------------------
# VECTOR DB
# --------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
docs = splitter.create_documents(csv_texts)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def normalize_text(text):
    return re.sub(r"[^a-zA-Z]", "", str(text)).lower()

ALIASES = {
    "bengaluru": "bangalore",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "trivandrum": "thiruvananthapuram",
    "pondicherry": "puducherry",
    "Keralam": "Kerala",
    "Cochin": "Kochi",
    "Calicut": "Kozhikode",
    "Baroda": "Vadodara"
}
'''ALLOWED_DOMAINS = [
    "tripadvisor.com",
    "makemytrip.com",
    "incredibleindia.org",
    "yatra.com",
    "holidify.com",
    "thrillophilia.com",
    "fabhotels.com",
    "agoda.com",
    "booking.com",
    "lonelyplanet.com",
    "traveltriangle.com",
    "goibibo",
]'''


def validate_location_from_csv(user_input):

    user_clean = normalize_text(user_input)

    # alias check
    if user_clean in ALIASES:
        return True, ALIASES[user_clean]

    normalized_map = {
        normalize_text(loc): loc for loc in VALID_LOCATIONS
    }

    # EXACT match first
    if user_clean in normalized_map:
        return True, normalized_map[user_clean]

    # WORD-LEVEL matching
    words = user_clean.split()

    for word in words:
        if word in normalized_map:
            return True, normalized_map[word]

    # FUZZY match ONLY if similarity is high
    match, score, _ = process.extractOne(
        user_clean,
        normalized_map.keys(),
        scorer=fuzz.ratio
    )

    #print("FUZZY:", user_clean, "->", match, score)

    # STRICT threshold
    if score >= 90:
        return True, normalized_map[match]

    return False, None

def extract_city_from_query(text):
    text_clean = normalize_text(text)

    normalized_map = {
        normalize_text(loc): loc for loc in VALID_LOCATIONS
    }

    for loc_clean, original in normalized_map.items():
        if loc_clean in text_clean:
            return original

    return None

def extract_trip_details(text):

    text = text.lower()

    days = None

    # -----------------------------------
    # BLOCK MONTH/YEAR
    # -----------------------------------
    if re.search(r"\b(month|months|year|years)\b", text):
        return None, None, (
            "Only short trips up to 7 days are supported."
        )

    # -----------------------------------
    # DIGIT + DAYS
    # Example: 3 days
    # -----------------------------------
    day_digit_match = re.search(
        r"(\d+)\s*(day|days)",
        text
    )

    if day_digit_match:
        days = int(day_digit_match.group(1))

    # -----------------------------------
    # DIGIT + WEEKS
    # Example: 1 week
    # -----------------------------------
    week_digit_match = re.search(
        r"(\d+)\s*(week|weeks)",
        text
    )

    if week_digit_match:
        days = int(week_digit_match.group(1)) * 7

    # -----------------------------------
    # WORD + DAYS/WEEKS
    # Example:
    # two days
    # one week
    # -----------------------------------
    if days is None:

        word_match = re.search(
            r"([a-z\s-]+)\s*(day|days|week|weeks)",
            text
        )

        if word_match:

            try:

                number_part = word_match.group(1).strip()

                value = w2n.word_to_num(number_part)

                unit = word_match.group(2)

                if "week" in unit:
                    days = value * 7
                else:
                    days = value

            except:
                pass

    # -----------------------------------
    # NATURAL LANGUAGE
    # -----------------------------------
    if days is None:

        if "a day" in text:
            days = 1

        elif "a week" in text:
            days = 7

    # -----------------------------------
    # MAX LIMIT
    # -----------------------------------
    if days and days > 7:

        return None, None, (
            "Only trips up to 7 days are supported."
        )

    # -----------------------------------
    # BUDGET
    # -----------------------------------
    budget_match = re.search(
        r"(?:₹|rs|rupees)?\s*(\d{3,7})",
        text
    )

    budget = int(budget_match.group(1)) if budget_match else None

    return days, budget, None

def web_search(query: str, max_results: int = 5):
    try:
        res = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )

        results = []
        for r in res.get("results", []):
            results.append({
                "title": r.get("title"),
                "body": r.get("content"),
                "url": r.get("url")
            })
        print(f"[Tavily] Got {len(results)} results for: {query}") 
        return results

    except Exception as e:
        print("Search error:", e)
        return []
    
def query_mentions_location(text):
    """Returns True if the query contains a recognizable city/state."""
    text_clean = normalize_text(text)
    normalized_map = {normalize_text(loc): loc for loc in VALID_LOCATIONS}
    
    # Check aliases
    if text_clean in ALIASES:
        return True
    
    # Check fuzzy match (same threshold as validate_location_from_csv)
    match, score, _ = process.extractOne(
        text_clean,
        normalized_map.keys(),
        scorer=fuzz.ratio
    )
    return score >= 75    
  

# --------------------------------------------------
# STATE
# --------------------------------------------------
class TravelState(TypedDict, total=False):
    user_input: str
    cleaned_input: str
    intent: str
    city: str
    location_confirmed: bool
    draft: str
    response: str


# --------------------------------------------------
# NODES
# --------------------------------------------------
def normalize(state: TravelState):
    txt = re.sub(r"\s+", " ", state["user_input"].strip())
    return {"cleaned_input": txt}


def validate_location_node(state: TravelState):
    text = state["cleaned_input"]

    # Step 1: full sentence match
    is_valid, loc = validate_location_from_csv(text)
    if is_valid:
        return {"city": loc, "location_confirmed": True}

    # Step 2: try bigrams + single words (reversed = check end of sentence first)
    words = text.lower().split()
    
    candidates = []
    
    # single words
    candidates += [words[i] for i in range(len(words))]
    
    # bigrams (two adjacent words joined)
    candidates += [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    
    # trigrams (three adjacent words joined — covers "jammu and kashmir" etc.)
    candidates += [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]

    # check from end of sentence first (location usually mentioned last)
    for candidate in reversed(candidates):
        is_valid, loc = validate_location_from_csv(candidate)
        if is_valid:
            return {"city": loc, "location_confirmed": True}

    # Step 3: already confirmed from session
    if state.get("location_confirmed"):
        return {}

    # Step 4: invalid
    #return {"response": "Please enter a valid Indian city or state to begin."}
    return {
    "city": None,
    "location_confirmed": False,
    "response": (
        "This chatbot supports only Indian destinations.\n\n"
        "Please enter a valid Indian city or state."
    )
}
def classify_intent(state: TravelState):
    text = state["cleaned_input"].lower()

    #  expanded weather keywords
    if any(w in text for w in [
        "weather", "temperature", "climate", "forecast", "rain", "hot", "cold"
    ]):
        return {"intent": "weather"}

    elif any(w in text for w in ["plan", "trip", "itinerary","budget", "travel"]):
        return {"intent": "plan"}

    else:
        return {"intent": "rag"}

# --------------------------------------------------
# WEATHER NODE
# --------------------------------------------------
def weather_node(state: TravelState):
    api_key = os.getenv("WEATHER_API_KEY")

    query_city = extract_city_from_query(state["cleaned_input"])
    city = query_city if query_city else state.get("city")

    if not api_key or not city:
        return {"draft": "Weather service is not configured properly."}

    url = "https://api.weatherapi.com/v1/current.json"  # use HTTPS

    try:
        res = requests.get(
            url,
            params={"key": api_key, "q": city},
            timeout=8   # increased timeout
        )

        res.raise_for_status()  # catch HTTP errors
        data = res.json()

        if "error" in data:
            return {"draft": data["error"]["message"]}

        return {
            "draft": f"{data['location']['name']}: {data['current']['temp_c']}°C, {data['current']['condition']['text']}"
        }

    except requests.exceptions.Timeout:
        return {"draft": "Weather service is taking too long. Please try again."}

    except requests.exceptions.RequestException:
        return {"draft": "Unable to fetch weather right now. Please try later."}

# --------------------------------------------------
# RAG NODE
# --------------------------------------------------
def rag_node(state: TravelState):
    user_query = state["cleaned_input"]
    selected_loc = state.get("city")

    if not selected_loc:
        return {"draft": "Please provide a valid Indian city or state."}

    #  FILTER DATAFRAME FIRST
    filtered_df = df[
    df[CITY_COL].apply(lambda x: normalize_text(str(x))).str.contains(normalize_text(selected_loc), na=False) |
    df[STATE_COL].apply(lambda x: normalize_text(str(x))).str.contains(normalize_text(selected_loc), na=False)]

    if filtered_df.empty:
        return {"draft": f"No data found for {selected_loc.title()}."}

    # convert filtered rows to text
    rows = []
    for _, row in filtered_df.iterrows():
        text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        rows.append(text)

    #  create temporary retriever ONLY for that location
    docs = splitter.create_documents(rows)
    temp_vector = FAISS.from_documents(docs, embedding_model)
    temp_retriever = temp_vector.as_retriever(search_kwargs={"k": 2})

    docs = temp_retriever.invoke(user_query)
    ###########
    #print("\n===== RETRIEVED DOCS =====")

    #for d in docs:
     #print(d.page_content[:500])
     #print("-------------------")
    ############

    context = "\n\n".join([d.page_content[:300] for d in docs])

    prompt = f"""

You are an Indian tourism assistant.

Answer ONLY about {selected_loc}.

If the answer is not available in context,
say:
"Information not available."

Keep answers concise and tourism-focused.

Context:
{context}

Question: {user_query}
"""

    res = model.invoke(prompt)

    return {"draft": res.content}

# --------------------------------------------------
# PLAN NODE
# --------------------------------------------------
def plan_node(state: TravelState):

    city = state.get("city")
    user_query = state["cleaned_input"]

    if not city:
        return {
            "draft": "Please select a valid Indian location first."
        }
    if city.lower() not in [loc.lower() for loc in VALID_LOCATIONS]:
     return {
        "draft": "Only Indian destinations are supported."
    }
    # -----------------------------------------
    # Extract trip details
    # -----------------------------------------
    days, budget, duration_error = extract_trip_details(user_query)

    if duration_error:
      return {
        "draft": duration_error
      }

    days_text = f"{days} day trip" if days else "travel trip"
    budget_text = f"under ₹{budget}" if budget else ""

    # -----------------------------------------
    # Build smarter search query
    # -----------------------------------------
    search_query = (
        f"{days_text} itinerary for {city} "
        f"best places hotels food transport budget {budget_text}"
    )

    # -----------------------------------------
    # Web Search
    # -----------------------------------------
    try:
        results = web_search(search_query, max_results=3)

    except Exception as e:

        print("Search Error:", e)

        results = []

    # -----------------------------------------
    # FALLBACK IF SEARCH FAILS
    # -----------------------------------------
    if not results:

        fallback_prompt = f"""
        Create a detailed travel plan for {city}, India.

        Include:
        - day-wise itinerary
        - tourist attractions
        - famous foods
        - hotel suggestions
        - estimated budget
        - local transport tips

        Keep response clean and structured.
        """

        try:

            res = model.invoke(fallback_prompt)

            return {
                "draft": (
                    f" Travel Plan for {city.title()}\n\n"
                    f"{res.content}"
                )
            }

        except Exception as e:

            return {
                "draft": "Unable to generate travel plan right now."
            }

    # -----------------------------------------
    # Convert search results to text
    # -----------------------------------------
    search_text = ""
    useful_links = []

    for r in results:

        title = r.get("title", "No title")
        body = r.get("body", "")[:300]
        url = r.get("url", "")

        search_text += f"""
    Title: {title}
    Description: {body}
    URL: {url}

    """

        if url and url.startswith("http"):

           useful_links.append(
            f"🔗 {title}\n{url}"
           )
        #print("URL FROM TAVILY:", url)    


    # -----------------------------------------
    # LLM Prompt for clean formatting
    # -----------------------------------------
    prompt = f"""
You are a strict Indian tourism planner.

IMPORTANT:
- Generate itinerary ONLY for {city}
- Never mention another city
- Never generate generic plans
- If information is limited, still stay focused on {city}

Generate a CLEAN and WELL-FORMATTED travel itinerary.

STRICT RULES:
- No markdown symbols like ** or ##
- No asterisks *
- No plus signs +
- Use simple clean text
- Keep response concise
- Use emojis for sections
- Organize properly for chatbot display
- Avoid repetition
- Mention approximate hotel budget
- Mention famous foods
- Mention major attractions
- Mention transport tips if useful
- Put every point on a NEW LINE
- Add blank line between sections
- Format neatly for HTML chatbot display
- Generate EXACTLY {days if days else 3} days itinerary
- Do NOT generate more than {days if days else 3} days
- Each day must have unique places
- Stop after Day {days if days else 3}

FORMAT:

📅 Day 1
- place
- place

🍴 Food to Try
- food items

🏨 Suggested Stay
- hotel names with approx budget

💰 Estimated Budget
- Budget range

🚕 Travel Tips
- short tips

User Request:
{user_query}

Location:
{city}

Trip Duration:
{days if days else 'Not specified'}

Budget:
{budget if budget else 'Not specified'}

Web Search Data:
{search_text}
""" 
    MAX_DAYS = 7
    days_notice = ""
    if days:
      if days < 1:
        days = 1
      elif days > MAX_DAYS:
        days = MAX_DAYS
        days_notice = (f"Maximum supported itinerary is {MAX_DAYS} days.\n\n")

    # -----------------------------------------
    # Generate final formatted response
    # -----------------------------------------
    try:

        res = model.invoke(prompt)

        final_response = f"""
        {days_notice}
🌍 Travel Plan for {city.title()}

{res.content}
"""

        # -----------------------------------------
        # Add useful links
        # -----------------------------------------
        if useful_links:

            final_response += "\n\n🔗 Useful Links\n\n"

            for i, link in enumerate(useful_links[:3], 1):

                final_response += f"{i}. {link}\n\n"
        #print("USEFUL LINKS:", useful_links)
        #print(final_response)
        return {
            "draft": final_response
        }

    except Exception as e:

        return {
            "draft": (
                f"Unable to generate travel plan right now. "
                f"Error: {str(e)}"
            )
        }
# --------------------------------------------------
# FINALIZE
# --------------------------------------------------
def finalize(state: TravelState):
    if state.get("response"):
        return {"response": state["response"]}
    if state.get("city") and state["city"] not in state.get("draft", "").lower():
        return {"response": f"Showing results for {state['city'].title()} only.\n\n{state.get('draft')}"}
    return {
        "response": state.get("draft", "This chatbot supports only Indian cities and states.")
    }

# --------------------------------------------------
# ROUTER
# --------------------------------------------------
def route(state: TravelState):
    if state.get("response") or not state.get("location_confirmed"):
        return "finalize"
    return state["intent"]  # "weather", "plan", or "rag"

# --------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------
builder = StateGraph(TravelState)

builder.add_node("normalize", normalize)
builder.add_node("validate_location", validate_location_node)
builder.add_node("classify_intent", classify_intent)

builder.add_node("weather", weather_node)
builder.add_node("rag", rag_node)
builder.add_node("plan", plan_node)
builder.add_node("finalize", finalize)

builder.add_edge(START, "normalize")
builder.add_edge("normalize", "validate_location")
builder.add_edge("validate_location", "classify_intent")

builder.add_conditional_edges("classify_intent", route)

builder.add_edge("weather", "finalize")
builder.add_edge("rag", "finalize")
builder.add_edge("plan", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()

# --------------------------------------------------
# FLASK API
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    
    # --------------------------------------------------
    # GREETINGS
    # --------------------------------------------------
    welcome_words = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening"
    ]

    text = user_msg.lower().strip()

    if text in welcome_words:

    # keep session but reset context lightly
     #session.setdefault("travel_context", {})

     return jsonify({
        "response": (
            "👋 Hello! Welcome to Indian Tourism Chatbot.\n\n"
            "I can help you with:\n"
            "- Travel plans for Indian cities\n"
            "- Weather updates\n"
            "- Places to visit\n\n"
            "Just tell me a State like 'Goa' or 'Kerala' or cities like Kochi, Ajmer to begin!"
        )
    })

    # --------------------------------------------------
    # FAREWELLS
    # --------------------------------------------------
    exit_words = [
    "bye",
    "goodbye",
    "see you",
    "thank you",
    "thanks"
     ]

  

    if text in exit_words:

     session["travel_context"] = {
        "city": None,
        "location_confirmed": False
    }


     return jsonify({
        "response": "Thank you for using Indian Tourism Chatbot!"
    })

    ############
    # --------------------------------------------------
    # TRAVEL KEYWORDS 
    # --------------------------------------------------
   
    travel_keywords = [

    # planning
    "plan",
    "trip",
    "travel",
    "itinerary",
    "vacation",
    "tour",
    "holiday",

    # duration
    "day",
    "days",
    "week",
    "weeks",

    # money
    "budget",
    "cost",
    "price",
    "expense",

    # weather
    "weather",
    "temperature",
    "climate",
    "forecast",
    "rain",
    "hot",
    "cold",

    # stay
    "hotel",
    "stay",
    "resort",
    "hostel",
    "accommodation",

    # transport
    "transport",
    "taxi",
    "metro",
    "bus",
    "train",
    "flight",

    # tourism
    "places",
    "tourist",
    "destination",
    "beach",
    "hill",
    "temple",
    "food",

    # follow-up conversational words
    "there",
    "nearby",
    "around",
    "local",
    "famous",
    "visit",
    "where"
     ]

    # Clear session ONLY if completely unrelated query
    if not query_mentions_location(user_msg):

     if not any(k in text for k in travel_keywords):

        session["travel_context"] = {
            "city": None,
            "location_confirmed": False
        }
    #---------------------------------
    # # NEW BLOCK
    # Clear stale session only for NEW invalid travel destination requests

    new_trip_keywords = [
    "plan",
    "trip",
    "travel",
    "itinerary",
    "vacation",
    "tour",
    "holiday"
]

    #if any(k in user_msg.lower() for k in new_trip_keywords):
    if any(k in text for k in new_trip_keywords):

      mentioned_city = extract_city_from_query(user_msg)

      if not mentioned_city:

        session.pop("city", None)
        session.pop("location_confirmed", None)  
    #############
   
    # --------------------------------------------------
    # HARD BLOCK — off-topic words
    # --------------------------------------------------
    general_block_words = [
    "chatbot",
    "python",
    "president",
    "actor",
    "cricket",
    "movie",
    "technology",
    "ai"
    ]

    #if any(w in user_msg.lower() for w in general_block_words):
    if any(w in text for w in general_block_words):

     return jsonify({
        "response":
        "I can help only with Indian tourism, travel planning, and weather information."
    })
    ############ 
   
    # --------------------------------------------------
    # INVOKE GRAPH
    # --------------------------------------------------
    memory = session.get("travel_context", {})

    state = {
    "user_input": user_msg,
    "location_confirmed": memory.get("location_confirmed", False),
    "city": memory.get("city")
         }

    result = graph.invoke(state)
    #------------
    # Create travel memory
    city = result.get("city")
    loc_confirmed = result.get("location_confirmed", False)

    if city and loc_confirmed:
      session["travel_context"] = {
        "city": city,
        "location_confirmed": True
    }
    else:
    #  safety cleanup
     session.pop("travel_context", None)
    #-----------
    #session["location_confirmed"] = result.get("location_confirmed", session.get("location_confirmed"))
    #session["city"] = result.get("city", session.get("city"))
    #session.pop("travel_context", None)
    
    return jsonify({"response": result.get("response")})



# --------------------------------------------------
# RUN APP
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)