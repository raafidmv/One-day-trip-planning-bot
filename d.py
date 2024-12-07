import os
import re
import json
import requests
from typing import List, Dict, Any

import openai
import streamlit as st
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

class TripPlannerAssistant:
    def __init__(self):
        # Initialize OpenAI API
        self.setup_openai_api()
        
        # Initialize session state
        self.initialize_session_state()
        
        # Initialize LangChain components
        self.initialize_langchain()

    def setup_openai_api(self):
        """Set up OpenAI API key from environment or user input"""
        # Try to get API key from environment variable
        api_key = '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'  # Replace with your actual API key

        # If no API key in environment, prompt user to input
        if not api_key:
            api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        
        # Validate and set API key
        if api_key:
            try:
                openai.api_key = api_key
                os.environ['OPENAI_API_KEY'] = api_key
            except Exception as e:
                st.error(f"Error setting up OpenAI API: {e}")

    def initialize_session_state(self):
        """Initialize or reset the session state"""
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "system", "content": "You are an advanced one-day trip planning AI assistant. "
                 "Help users create personalized, detailed itineraries for their day trips."}
            ]
        
        # User context storage
        if "user_context" not in st.session_state:
            st.session_state["user_context"] = {
                "destination": None,
                "date": None,
                "budget": None,
                "interests": [],
                "start_location": None,
                "persona": None,
                "weather": None
            }
        
        # Memory graph for user preferences
        if "memory_graph" not in st.session_state:
            st.session_state["memory_graph"] = {
                "triplets": [],  # Entity-Relationship-Entity
                "user_profiles": {}
            }
        
        # Current itinerary storage
        if "current_itinerary" not in st.session_state:
            st.session_state["current_itinerary"] = []

    def initialize_langchain(self):
        """Initialize LangChain components"""
        # Initialize memory
        memory = ConversationBufferMemory(return_messages=True)
        
        # Initialize ChatOpenAI model
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Create conversation chain
        prompt_template = PromptTemplate(
            input_variables=["history", "input"], 
            template="""You are an AI trip planner. Based on the conversation history and the latest input, 
            help the user plan a personalized one-day trip. 
            Conversation History:
            {history}
            
            Latest Request:
            {input}
            
            Your detailed, personalized response:"""
        )
        
        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt_template,
            verbose=False
        )
        
        st.session_state["conversation_chain"] = conversation_chain

    def fetch_weather(self, city, date):
        """Fetch weather information for the destination"""
        try:
            # Replace with a real weather API call 
            # This is a placeholder implementation
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a weather information assistant."},
                    {"role": "user", "content": f"Provide a brief weather forecast for {city} on {date}. Include temperature, conditions, and recommendations."}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could not fetch weather: {e}"

    def extract_context(self, user_input):
        """Extract context from user input"""
        context = st.session_state["user_context"]
        
        # Destination extraction
        destination_match = re.search(r'\b(in|to)\s+([A-Za-z\s]+)', user_input, re.IGNORECASE)
        if destination_match:
            context["destination"] = destination_match.group(2).strip()
        
        # Date extraction
        date_match = re.search(r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))', user_input, re.IGNORECASE)
        if date_match:
            context["date"] = date_match.group(1)
            # Fetch weather for the date
            if context["destination"]:
                context["weather"] = self.fetch_weather(context["destination"], context["date"])
        
        # Budget extraction
        budget_match = re.search(r'\$(\d+)', user_input)
        if budget_match:
            context["budget"] = int(budget_match.group(1))
        
        # Interests extraction
        interest_keywords = {
            "historical": ["history", "historical", "ancient", "heritage"],
            "cultural": ["culture", "art", "museum", "gallery"],
            "food": ["food", "cuisine", "restaurant", "eating"],
            "nature": ["nature", "park", "outdoor", "landscape"],
            "adventure": ["adventure", "exciting", "thrill"],
            "shopping": ["shopping", "shops", "market"]
        }
        
        for category, keywords in interest_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                if category not in context["interests"]:
                    context["interests"].append(category)
        
        return context

    def generate_persona(self):
        """Generate a user persona based on collected preferences"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a persona generation assistant."},
                    {"role": "user", "content": f"Create a detailed persona based on these preferences: {st.session_state['user_context']}"}
                ]
            )
            persona = response.choices[0].message.content
            st.session_state["user_context"]["persona"] = persona
            return persona
        except Exception as e:
            return f"Could not generate persona: {e}"

    def generate_itinerary(self):
        """Generate a comprehensive itinerary based on user context"""
        context = st.session_state["user_context"]
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert trip planner. Generate a detailed, optimized one-day itinerary."},
                    {"role": "user", "content": f"""
                    Generate a comprehensive one-day itinerary for {context.get('destination', 'the city')} with the following constraints:
                    - Interests: {', '.join(context.get('interests', []))}
                    - Budget: ${context.get('budget', 'Not specified')}
                    - Date: {context.get('date', 'Not specified')}
                    - Weather: {context.get('weather', 'Not available')}
                    
                    Please provide:
                    1. A list of 3-4 attractions
                    2. Estimated time at each location
                    3. Transportation between locations
                    4. Estimated costs
                    5. Any weather-related recommendations
                    """}
                ]
            )
            
            itinerary_text = response.choices[0].message.content
            
            # Parse the itinerary (you might want more robust parsing)
            st.session_state["current_itinerary"] = itinerary_text
            return itinerary_text
        
        except Exception as e:
            return f"Error generating itinerary: {e}"

    def update_memory_graph(self, entity1, relationship, entity2):
        """Update the memory graph with new triplets"""
        memory_graph = st.session_state["memory_graph"]
        memory_graph["triplets"].append((entity1, relationship, entity2))

    def process_user_input(self, user_input):
        """Process user input and generate appropriate response"""
        # Extract context
        self.extract_context(user_input)
        
        # Use conversation chain to generate response
        conversation_chain = st.session_state["conversation_chain"]
        
        try:
            response = conversation_chain.predict(input=user_input)
            
            # Update messages
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})
            
            return response
        
        except Exception as e:
            return f"An error occurred: {e}"

    def render_ui(self):
        """Render the Streamlit UI"""
        st.title("🌍 Personalized One-Day Trip Planner")
        
        # Display chat messages
        for message in st.session_state["messages"]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # Sidebar for context and itinerary
        st.sidebar.write("### 📍 Current Context")
        st.sidebar.json(st.session_state["user_context"])
        
        st.sidebar.write("### 📅 Current Itinerary")
        st.sidebar.write(st.session_state.get("current_itinerary", "No itinerary generated yet"))
        
        # Chat input
        if prompt := st.chat_input("What's your trip plan?"):
            with st.chat_message("user"):
                st.write(prompt)
            
            # Special commands
            if "generate persona" in prompt.lower():
                persona = self.generate_persona()
                st.chat_message("assistant").write(f"Generated Persona:\n{persona}")
            elif "create itinerary" in prompt.lower():
                itinerary = self.generate_itinerary()
                st.chat_message("assistant").write(f"Generated Itinerary:\n{itinerary}")
            else:
                # Regular conversation processing
                response = self.process_user_input(prompt)
                
                with st.chat_message("assistant"):
                    st.write(response)

def main():
    assistant = TripPlannerAssistant()
    assistant.render_ui()

if __name__ == "__main__":
    main()