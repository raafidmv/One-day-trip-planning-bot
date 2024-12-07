import os
import json
import streamlit as st
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env file
load_dotenv()

# Define the state of the tour planning process
class TourPlanningState(BaseModel):
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    itinerary: List[Dict[str, Any]] = Field(default_factory=list)
    current_step: str = Field(default="start")
    conversation_history: List[BaseMessage] = Field(default_factory=list)
    weather_recommendations: str = Field(default="")

# Main Tour Planning Agent
class TourPlanningAgent:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API Key must be set in .env file or environment variables")
        
        self.llm = ChatOpenAI(
            api_key=api_key, 
            model="gpt-3.5-turbo",  
            temperature=0.3
        )
    
    def generate_response(self, state: TourPlanningState) -> Dict[str, Any]:
        # Create a comprehensive prompt template for tour planning
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI travel assistant helping users plan a personalized one-day tour. 
            Your goal is to gather comprehensive information about their trip preferences through conversation.
            
            Key information to collect:
            1. Destination city
            2. Preferred date
            3. Budget
            4. Food preferences
            5. Personal interests and activities
            6. Any specific requirements or constraints

            Guide the conversation naturally, asking follow-up questions to refine the tour plan.
            Be friendly, helpful, and aim to create a tailored travel experience."""),
            MessagesPlaceholder(variable_name="conversation_history"),
            ("human", "{current_input}")
        ])
        
        # Prepare the conversation context
        current_input = state.user_preferences.get('current_input', '')
        messages = state.conversation_history + [
            HumanMessage(content=current_input)
        ]
        
        # Get the response from the LLM
        response = self.llm(
            prompt_template.format_messages(
                conversation_history=state.conversation_history,
                current_input=current_input
            )
        )
        
        # Extract city if mentioned
        user_preferences = state.user_preferences.copy()
        input_lower = current_input.lower()
        if any(city in input_lower for city in ['bangalore', 'bengaluru']):
            user_preferences['city'] = 'Bangalore'
        
        # Update conversation history and user preferences
        return {
            "conversation_history": messages + [AIMessage(content=response.content)],
            "user_preferences": user_preferences,
            "current_step": "conversing"
        }
    
    def generate_itinerary(self, state: TourPlanningState) -> Dict[str, Any]:
        # Create a prompt to generate a tour itinerary based on collected preferences
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Generate a detailed one-day tour itinerary based on the following user preferences:
            
            City: {city}
            Date: {date}
            Budget: {budget}
            Interests: {interests}
            Food Preferences: {food_preferences}
            
            Create a comprehensive plan that includes:
            - Specific attractions and sites to visit
            - Estimated timings
            - Dining recommendations
            - Transportation suggestions
            - Budget-friendly options"""),
            ("human", "Please create my personalized tour plan")
        ])
        
        # Prepare the prompt with collected preferences
        prompt = prompt_template.format_messages(
            city=state.user_preferences.get('city', 'Unknown'),
            date=state.user_preferences.get('date', 'Not specified'),
            budget=state.user_preferences.get('budget', 'Not specified'),
            interests=state.user_preferences.get('interests', 'Not specified'),
            food_preferences=state.user_preferences.get('food_preferences', 'Not specified')
        )
        
        # Get the itinerary from the LLM
        response = self.llm(prompt)
        
        return {
            "itinerary": [{"description": response.content}],
            "current_step": "itinerary_generated"
        }

# Function to create and compile the tour planning workflow graph
def create_tour_planning_graph():
    workflow = StateGraph(TourPlanningState)
    
    tour_agent = TourPlanningAgent()
    
    workflow.add_node("generate_response", tour_agent.generate_response)
    workflow.add_node("generate_itinerary", tour_agent.generate_itinerary)
    
    workflow.set_entry_point("generate_response")
    
    # Add edges with a condition
    workflow.add_conditional_edges(
        "generate_response", 
        lambda state: "generate_itinerary" if state.user_preferences.get('ready_for_itinerary') else "generate_response"
    )
    workflow.add_edge("generate_itinerary", END)
    
    return workflow.compile()

# Streamlit Application
def streamlit_tour_planner():
    # Set page configuration
    st.set_page_config(page_title="AI Tour Planner Chatbot", page_icon="✈️", layout="wide")
    
    # Initialize session state for conversation and workflow
    if 'tour_planning_graph' not in st.session_state:
        st.session_state.tour_planning_graph = create_tour_planning_graph()
    
    # Initialize conversation state with initial message
    if 'conversation_state' not in st.session_state:
        # Create initial system and AI messages
        initial_system_message = HumanMessage(content="I am a travel assistant ready to help you plan your perfect tour.")
        initial_ai_message = AIMessage(content="Hello! I'm your AI travel assistant. I'm excited to help you plan an incredible one-day tour. What destination are you interested in exploring?")
        
        st.session_state.conversation_state = TourPlanningState(
            conversation_history=[initial_system_message, initial_ai_message]
        )
    
    # Title and introduction
    st.title("🌍 AI Tour Planner Chatbot")
    st.write("Chat with our AI assistant to plan your perfect one-day city tour!")
    
    # Chat message container
    chat_container = st.container()
    
    # Chat history display
    with chat_container:
        for message in st.session_state.conversation_state.conversation_history:
            if isinstance(message, HumanMessage):
                st.chat_message("user").write(message.content)
            elif isinstance(message, AIMessage):
                st.chat_message("assistant").write(message.content)
    
    # Chat input
    user_input = st.chat_input("Tell me about your dream tour...")
    
    # Process user input
    if user_input:
        # Display user message
        st.chat_message("user").write(user_input)
        
        # Update conversation state
        st.session_state.conversation_state.user_preferences['current_input'] = user_input
        
        # Run the workflow to generate response
        with st.spinner("Thinking..."):
            try:
                # Invoke the workflow
                result = st.session_state.tour_planning_graph.invoke(st.session_state.conversation_state)
                
                # Update conversation state
                st.session_state.conversation_state = result
                
                # Display AI response
                if result.conversation_history:
                    st.chat_message("assistant").write(
                        result.conversation_history[-1].content
                    )
                
                # Check if itinerary is ready
                if result.current_step == 'itinerary_generated':
                    st.success("Tour Plan Generated!")
                    st.write("### Your Personalized Tour Itinerary:")
                    st.write(result.itinerary[0]['description'])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Main function to run the Streamlit app
def main():
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        st.error("Please set the OPENAI_API_KEY in your .env file or environment variables.")
        return
    
    # Run Streamlit app
    streamlit_tour_planner()

# Run the Streamlit application
if __name__ == "__main__":
    main()