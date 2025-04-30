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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CONFIGURATION
## AZURE SEARCH
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

## MODELS GPT4O
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

# RETRIVALS
def retrieve_martindale(query: list[dict], top_k: int = 10) -> str:
	logger.info(f"Martindale search for: {query}")
	try:
		results = MARTINDALE_SEARCH_CLIENT.search(
			search_text=query,
			top=top_k
		)
		documents = [doc.get("content") for doc in results]
		return "\n-----\n".join(documents)
	except Exception as e:
		logger.error(f"Martindale search error: {e}")
		return "Error: Unable to access Martindale database."

def retrieve_dailymed(query: str, top_k: int = 10) -> str:
	logger.info(f"DailyMed search for: {query}")
	try:
		results = DAILYMED_SEARCH_CLIENT.search(
			search_text=query,
			top=top_k
		)
		documents = [f"description : {doc.get('description')}\ningredient : {doc.get('ingredients')}" for doc in results]
		return "\n-----\n".join(documents)
	except Exception as e:
		logger.error(f"DailyMed search error: {e}")
		return "Error: Unable to access DailyMed database."

def retrieve_fda(query: str, top_k: int = 10) -> str:
	logger.info(f"FDA research for: {query}")
	try:
		results = FDA_SEARCH_CLIENT.search(
			search_text=query,
			top=top_k
		)
		documents = [f"description : {doc.get('descriptions')}\nactive_ingredients : {doc.get('active_ingredients')}\nnotice / {doc.get('notice')}" for doc in results]
		return "\n-----\n".join(documents)
	except Exception as e:
		logger.error(f"FDA search error: {e}")
		return "Error: Unable to access FDA database."

# AGENTS
# AGENT RETRIVALS
class CustomAssistantAgent(agentchat.AssistantAgent):
	def __init__(self, name, pre_fn=None, **kwargs):
		super().__init__(name=name, **kwargs)
		self.pre_fn = pre_fn

	def generate_reply(self, messages=None, sender=None, config=None):
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

MARTINDALE_AGENT = CustomAssistantAgent(
	name="Agent_Martindale",
	pre_fn=retrieve_martindale,
	llm_config={"config_list": [AZURE_OPENAI_CONFIG_GPT4O1]},
	system_message=(
		"You are a specialized agent who consults only the Martindale database.\n"
		"You must research the information requested and present Martindale's results clearly.\n"
		"Always identify yourself as 'Martindale Agent' and only present information from this source.\n"
		"Mention clearly when you can't find relevant information in your database.\n"
		"Translate the answer into the user's language."
	),
)

DAILYMED_AGENT = CustomAssistantAgent(
	name="Agent_DailyMed",
	pre_fn=retrieve_dailymed,
	llm_config={"config_list": [AZURE_OPENAI_CONFIG_GPT4O1]},
	system_message=(
		"You are a specialized agent who consults only the DailyMed database.\n"
		"You must research the information requested and present DailyMed's results clearly.\n"
		"Always identify yourself as 'DailyMed Agent' and only present information from this source.\n"
		"Mention clearly when you can't find relevant information in your database.\n"
		"Translate the answer into the user's language."
	)
)

FDA_AGENT = CustomAssistantAgent(
	name="Agent_FDA",
	pre_fn=retrieve_fda,
	llm_config={"config_list": [AZURE_OPENAI_CONFIG_GPT4O1]},
	system_message=(
		"You are a specialized agent who consults only the FDA (Food and Drug Administration) database.\n"
		"You must research the information requested and present FDA (Food and Drug Administration)'s results clearly.\n"
		"Always identify yourself as 'FDA (Food and Drug Administration) Agent' and only present information from this source.\n"
		"Mention clearly when you can't find relevant information in your database.\n"
		"Translate the answer into the user's language."
	),
) 

# AGENT WRITER
WRITER_AGENT = agentchat.AssistantAgent(
	name="Editor_Agent",
	llm_config={"config_list": [AZURE_OPENAI_CONFIG_GPT4O]},
	system_message=(
		"You are a medical editor who synthesizes information provided by other agents.\n"
		"Wait until the source agents (Martindale, DailyMed and FDA) have provided their information before responding.\n"
		"Your role is to create a clear, structured and easy-to-understand response based solely on the information provided.\n"
		"Clearly identify the source of each piece of information in your final response.\n"
		"If the sources contradict each other, mention this and explain the differences.\n"
		"Translate the answer into the user's language."
	)
)

# AGENT ORCHESTRATOR
class ResponseCaptureAgent(agentchat.AssistantAgent):
	def __init__(self, name, **kwargs):
		super().__init__(name=name, **kwargs)
		self.final_response = None
	
	def _process_received_message(self, message, sender, config):
		if message is not None and sender == "Editor_Agent":
			self.final_response = message
		return super()._process_received_message(message, sender, config)

OBSERVER_AGENT = ResponseCaptureAgent(
	name="Observer",
	llm_config={"config_list": [AZURE_OPENAI_CONFIG_GPT4O]},
	system_message="Tu es un observateur silencieux qui capture la réponse finale."
)

USER_PROXY_AGENT = agentchat.UserProxyAgent(
	name="User",
	human_input_mode="NEVER",
	max_consecutive_auto_reply=0,
	code_execution_config={"use_docker": False}
)

GROUP_CHAT = agentchat.GroupChat(
	agents=[USER_PROXY_AGENT, MARTINDALE_AGENT, DAILYMED_AGENT, FDA_AGENT, WRITER_AGENT, OBSERVER_AGENT],
	messages=[],
	max_round=10,
	speaker_selection_method="round_robin",
	allow_repeat_speaker=False
)

MANAGER_GROUP = agentchat.GroupChatManager(
	groupchat=GROUP_CHAT,
	llm_config={"config_list": [AZURE_OPENAI_CONFIG_GPT4O1]},
	system_message=(
		"You are the manager of this group conversation.\n"
		"Make sure the agents speak in this precise order after the initial question:\n"
		"1. Agent_Martindale - must research and present Martindale's information.\n"
		"2. Agent_DailyMed - must research and present DailyMed's information.\n"
		"3. Agent_FDA - must search and present FDA information\n"
		"4. Editor_Agent - to analyze information from both sources and present a summary\n"
		"5. End of conversation\n"
		"Don't let the agents talk in a different order, and make sure that each agent fulfills his or her role correctly.\n"
	)
)

@app.route("/api/medical-agents", methods=["POST"])
def medical_info_api():
	try:
		data = request.json
		if not data or "query" not in data:
			return jsonify({
				"status": "error",
				"error": "The query must contain a 'query' field."
			}), 400
		
		query = data["query"]
		conversation_history = data.get("conversation_history", [])

		if not isinstance(conversation_history, list):
			return jsonify({
				"status": "error",
				"error": "The 'conversation_history' field must be a list."
			}), 400

		formatted_history = []
		for msg in conversation_history[:10]:
			if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
				return jsonify({
					"status": "error", 
					"error": "Each message in the history must contain the fields 'role' and 'content'."
				}), 400
			
			formatted_history.append(f"User: {msg['content']}" if msg["role"] == "user" else f"Assistant: {msg['content']}")

		string_history = '\n'.join(formatted_history)
		conversation_context = f"Conversation history:\n {string_history}\n\nNew question: {query}" if formatted_history else query
		logger.info(f"Conversation context prepared with {len(formatted_history)} previous messages.")
		logger.info(f"Start of research for: {query} with history of {len(formatted_history)} messages")

		print(conversation_context)
		
		USER_PROXY_AGENT.initiate_chat(MANAGER_GROUP, message=conversation_context)

		writer_messages = [msg for msg in GROUP_CHAT.messages if msg["name"] == "Editor_Agent"]
		final_response = writer_messages[-1]["content"] if writer_messages else "No response generated."

		return jsonify({
			"status": "success",
			"response": final_response
		})

	except Exception as e:
		logger.error(f"API error: {traceback.format_exc()}")
		return jsonify({
			"status": "error",
			"error": str(e),
			"traceback": traceback.format_exc()
		}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
	logger.info("Functional API")
	return jsonify({"status": "API operational."}), 200

if __name__ == '__main__':
	app.run(host='0.0.0.0', port="8080", debug=False)