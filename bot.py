import telebot
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# --- 1. Setup Keys and APIs ---
BOT_TOKEN = "bot_father_token"
GEMINI_API_KEY = "your_gemini_api_key"

# Configure the Generative AI (Using the new google-genai library)
client = genai.Client(api_key=GEMINI_API_KEY)

bot = telebot.TeleBot(BOT_TOKEN)

# Configure the Vector Database
print("Loading Database and Embeddings...")
embed_model = SentenceTransformer('sentence-transformers/LaBSE')
db_client = chromadb.PersistentClient(path="./vector_store")
collection = db_client.get_collection(name="company_knowledge")

# --- The Memory Cache ---
response_cache = {}

# --- 2. Handle Messages ---
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_query = message.text
    print(f"\n--- New Message: {user_query} ---")
    
    # 1. CHECK THE CACHE FIRST
    if user_query in response_cache:
        print("Found in cache! Skipping database and AI.")
        bot.reply_to(message, response_cache[user_query])
        return
    
    # 2. Search the vector database
    query_vector = embed_model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=1)
    
    distance_score = results['distances'][0][0]
    raw_excel_answer = results['metadatas'][0][0]['answer']
    
    print(f"Database distance: {distance_score}")
    
    # Prevent bad guesses: If distance is too high, it's not in the Excel file
    if distance_score > 14.5:
        bot.reply_to(message, "I am sorry, but I couldn't find information regarding that in the company guidelines.")
        return

    # 3. The LLM Magic
    print("Asking Gemini to format the response...")
    
    prompt = f"""
    You are a polite and helpful HR assistant for a company. 
    The user asked: "{user_query}"
    
    Here is the official information retrieved from the company database:
    "{raw_excel_answer}"
    
    Instructions:
    1. Answer the user's question naturally using ONLY the provided company information.
    2. Do not invent any new rules or information.
    3. CRITICAL: You MUST reply in the exact same language that the user used in their question (e.g., if they asked in Burmese, you must reply in Burmese).
    """
    
    try:
        # Generate the smart response using the new SDK syntax
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        final_answer = response.text
        bot.reply_to(message, final_answer)
        print("Reply sent successfully!")
        
        # 4. SAVE TO CACHE FOR NEXT TIME
        response_cache[user_query] = final_answer
        
    except Exception as e:
        print(f"Error generating AI response: {e}")
        bot.reply_to(message, "I am having a little trouble connecting to my AI brain right now. Please try again in a minute!")

# --- 3. Start the Bot ---
print("Bot is online with full RAG AI pipeline and memory cache! Send a message.")
bot.infinity_polling()