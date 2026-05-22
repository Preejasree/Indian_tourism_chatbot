AI-POWERED INDIAN TOURISM CHATBOT

Project Name:
Indian Tourism Chatbot

Description:
This project is an AI-based tourism chatbot developed using Flask, LangGraph, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs). The chatbot helps users plan trips, get weather updates, retrieve tourism information, and generate personalized travel itineraries for Indian destinations.

========================================
FEATURES
Travel itinerary generation
Real-time weather updates
Tourism information retrieval
Budget-based travel planning
Indian city/state validation
Fuzzy matching for spelling correction
Retrieval-Augmented Generation (RAG)
FAISS vector database integration
Tavily web search integration
Session-based conversational memory
========================================
TECHNOLOGIES USED

Frontend:

HTML
CSS
JavaScript

Backend:

Python
Flask

AI & NLP:

LangGraph
Hugging Face Llama 3.1
Sentence Transformers

Database & Retrieval:

FAISS Vector Database
Pandas CSV Processing

APIs:

WeatherAPI
Tavily Search API

Libraries:

RapidFuzz
word2number
dotenv
========================================
PROJECT STRUCTURE

project_folder/
│
├── app.py
├── templates/
│ └── index.html
├── static/
│ ├── style.css
│ └── script.js
├── TopIndianPlacestoVisit_2.csv
├── .env
├── requirements.txt
└── README.txt

========================================
INSTALLATION STEPS


* Navigate to Project Folder

cd project_folder

* Create Virtual Environment

Windows:
python -m venv venv

* Activate Virtual Environment

Windows:
venv\Scripts\activate

* Install Required Packages

pip install -r requirements.txt

========================================
ENVIRONMENT VARIABLES

* Create a .env file and add:

HF_TOKEN=your_huggingface_token
TAVILY_API_KEY=your_tavily_api_key
WEATHER_API_KEY=your_weather_api_key

========================================
* RUN THE APPLICATION

Run the Flask application:

python app.py

Open browser:
http://127.0.0.1:5000

========================================
SUPPORTED USER QUERIES

Examples:

Plan a 3-day trip to Goa
Weather in Delhi
Best places to visit in Kerala
========================================
END OF FILE