from flask import Flask, request, jsonify
import traceback
import logging

from autogen import agentchat

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import azure.search.documents.models as models

from openai import AzureOpenAI

import os
from dotenv import load_dotenv
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# --- AZURE SEARCH CLIENT SETUP ---
try:
    MARTINDALE_SEARCH_CLIENT = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name="pyartes-martindale-index",
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )
    DAILYMED_SEARCH_CLIENT = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name="pyrates-dailymed-index",
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )
    FDA_SEARCH_CLIENT = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name="pyrates-fda-index",
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    )
except KeyError as e:
    logger.error(f"Missing environment variable: {e}. Check your .env file.")
    raise

# --- AZURE OPENAI CLIENT SETUP ---
try:
    AZURE_OPENAI_CONFIG_ADA = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_API_BASE_ADA"),
        api_key=os.getenv("AZURE_API_KEY_ADA"),
        api_version="2023-05-15"
    )
    AZURE_OPENAI_CONFIG_GPT4O1 = {
        "model": "pyrates-reasoning-o1",
        "api_key": os.getenv("AZURE_API_KEY"),
        "api_type": "azure",
        "base_url": os.getenv("AZURE_API_BASE"),
        "api_version": "2025-01-01-preview"
    }
    AZURE_OPENAI_CONFIG_GPT4O = {
        "model": "pyrates-text-gpt4o",
        "api_key": os.getenv("AZURE_API_KEY"),
        "api_type": "azure",
        "base_url": os.getenv("AZURE_API_BASE"),
        "api_version": "2024-12-01-preview"
    }
except KeyError as e:
    logger.error(f"Missing environment variable: {e}. Check your .env file.")
    raise

# --- RETRIEVAL FUNCTIONS ---
def retrieve_martindale(query: list[dict], top_k: int = 10) -> str:
    """
    Search the Martindale Azure Cognitive Search index.
    """
    logger.info(f"Martindale search for: {query}")
    try:
        results = MARTINDALE_SEARCH_CLIENT.search(search_text=query, top=top_k)
        documents = [doc.get("content") for doc in results]
        return "\n-----\n".join(documents)
    except Exception as e:
        logger.error(f"Martindale search error: {e}")
        return "Error: Unable to access Martindale database."

def retrieve_dailymed(query: str, top_k: int = 10) -> str:
    """
    Search the DailyMed Azure Cognitive Search index.
    """
    logger.info(f"DailyMed search for: {query}")
    try:
        results = DAILYMED_SEARCH_CLIENT.search(search_text=query, top=top_k)
        documents = [
            f"description : {doc.get('description')}\ningredient : {doc.get('ingredients')}"
            for doc in results
        ]
        return "\n-----\n".join(documents)
    except Exception as e:
        logger.error(f"DailyMed search error: {e}")
        return "Error: Unable to access DailyMed database."

def retrieve_fda(query: str, top_k: int = 10) -> str:
    """
    Search the FDA Azure Cognitive Search index.
    """
    logger.info(f"FDA research for: {query}")
    try:
        results = FDA_SEARCH_CLIENT.search(search_text=query, top=top_k)
        documents = [
            f"description : {doc.get('descriptions')}\nactive_ingredients : {doc.get('active_ingredients')}\nnotice / {doc.get('notice')}"
            for doc in results
        ]
        return "\n-----\n".join(documents)
    except Exception as e:
        logger.error(f"FDA search error: {e}")
        return "Error: Unable to access FDA database."

# --- CUSTOM ASSISTANT AGENT CLASS ---
class CustomAssistantAgent(agentchat.AssistantAgent):
    """
    A specialized AssistantAgent that uses a pre-processing function (pre_fn) to enrich the context.
    """
    def __init__(self, name, pre_fn=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.pre_fn = pre_fn

    def generate_reply(self, messages=None, sender=None, config=None):
        """
        Generates a reply after enriching the input using the pre_fn retrieval.
        """
        if messages is None:
            if sender is not None:
                user_messages = [msg for msg in self.chat_messages[sender] if msg["role"] == "user"]
                if not user_messages:
                    raise ValueError("No user messages found for sender.")
                messages = user_messages[-1]["content"]
            else:
                if isinstance(messages, list):
                    user_messages = [msg for msg in messages if msg["role"] == "user"]
                    if not user_messages:
                        raise ValueError("No user messages found in provided messages.")
                    messages = user_messages[-1]["content"]

        if self.pre_fn:
            enrichment = self.pre_fn(messages)
            if enrichment:
                new_message = {
                    "role": "system",
                    "content": f"Here's the knowledge you need to base your answer on: {enrichment}"
                }
                self.chat_messages[self].append(new_message)

        return super().generate_reply(sender=sender, config=config)

# --- RETRIEVAL AGENTS ---
MARTINDALE_AGENT = CustomAssistantAgent(...)
DAILYMED_AGENT = CustomAssistantAgent(...)
FDA_AGENT = CustomAssistantAgent(...)

# --- WRITER AGENT ---
WRITER_AGENT = agentchat.AssistantAgent(...)

# --- OBSERVER AGENT ---
class ResponseCaptureAgent(agentchat.AssistantAgent):
    """
    An agent that observes and stores the final response from the writer.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.final_response = None

    def _process_received_message(self, message, sender, config):
        if message is not None and sender == "Editor_Agent":
            self.final_response = message
        return super()._process_received_message(message, sender, config)

OBSERVER_AGENT = ResponseCaptureAgent(...)

USER_PROXY_AGENT = agentchat.UserProxyAgent(...)

# --- GROUP CHAT CONFIG ---
GROUP_CHAT = agentchat.GroupChat(...)
MANAGER_GROUP = agentchat.GroupChatManager(...)

# --- API ENDPOINT: MEDICAL MULTI-AGENT ---
@app.route("/api/medical-agents", methods=["POST"])
def medical_info_api():
    """
    Endpoint to handle multi-agent medical queries with optional conversation history.
    """
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"status": "error", "error": "The query must contain a 'query' field."}), 400

        query = data["query"]
        conversation_history = data.get("conversation_history", [])

        if not isinstance(conversation_history, list):
            return jsonify({"status": "error", "error": "The 'conversation_history' field must be a list."}), 400

        formatted_history = []
        for msg in conversation_history[:10]:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return jsonify({"status": "error", "error": "Each message must contain 'role' and 'content'."}), 400
            formatted_history.append(f"User: {msg['content']}" if msg["role"] == "user" else f"Assistant: {msg['content']}")

        string_history = '\n'.join(formatted_history)
        conversation_context = f"Conversation history:\n {string_history}\n\nNew question: {query}" if formatted_history else query

        logger.info(f"Start of research for: {query} with history of {len(formatted_history)} messages")
        USER_PROXY_AGENT.initiate_chat(MANAGER_GROUP, message=conversation_context)

        writer_messages = [msg for msg in GROUP_CHAT.messages if msg["name"] == "Editor_Agent"]
        final_response = writer_messages[-1]["content"] if writer_messages else "No response generated."

        return jsonify({"status": "success", "response": final_response})

    except Exception as e:
        logger.error(f"API error: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# --- API ENDPOINT: HEALTH CHECK ---
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    logger.info("Functional API")
    return jsonify({"status": "API operational."}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8080", debug=False)