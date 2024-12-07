import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Define the state of the tour planning process
class TourPlanningState(BaseModel):
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    itinerary: List[Dict[str, Any]] = Field(default_factory=list)
    current_step: str = Field(default="start")
    memory_graph: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[BaseMessage] = Field(default_factory=list)
    weather_recommendations: str = Field(default="")

# Agent to interact with the user and collect preferences
class UserInteractionAgent:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API Key must be set in .env file or environment variables")
        
        self.llm = ChatOpenAI(
            api_key=api_key, 
            model="gpt-3.5-turbo",  # Using a widely available model
            temperature=0.2
        )
        
    def collect_preferences(self, state: TourPlanningState) -> Dict[str, Any]:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful travel assistant collecting user preferences 
            for a one-day city tour. Guide the user through selecting:
            1. City to visit
            2. Date and time range
            3. Interests
            4. Budget
            5. Starting point"""),
            ("human", "{current_input}")
        ])
        
        # Prepare the prompt with current user input
        prompt = prompt_template.format_messages(current_input=state.user_preferences.get('current_input', ''))
        
        # Get the response from the LLM
        response = self.llm(prompt)
        
        # Update user preferences with collected information
        return {
            "user_preferences": {
                **state.user_preferences,
                "collected_preferences": response.content
            },
            "current_step": "preferences_collected"
        }

# Agent to generate the initial itinerary based on user preferences
class ItineraryGenerationAgent:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API Key must be set in .env file or environment variables")
        
        self.llm = ChatOpenAI(
            api_key=api_key, 
            model="gpt-3.5-turbo", 
            temperature=0.3
        )
        
    def generate_initial_itinerary(self, state: TourPlanningState) -> Dict[str, Any]:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Generate a one-day tour itinerary based on user preferences. 
            Provide a list of attractions with timings and brief details."""),
            ("human", "City: {city}, Interests: {interests}, Budget: {budget}")
        ])
        
        # Prepare the prompt with user preferences
        prompt = prompt_template.format_messages(
            city=state.user_preferences.get('city', ''),
            interests=", ".join(state.user_preferences.get('interests', [])),
            budget=state.user_preferences.get('budget', 0)
        )
        
        # Get the response from the LLM
        response = self.llm(prompt)
        
        # Update the itinerary with the generated content
        return {
            "itinerary": [{"description": response.content}],
            "current_step": "itinerary_generated"
        }

# Agent to optimize the itinerary based on budget and other constraints
class OptimizationAgent:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API Key must be set in .env file or environment variables")
        
        self.llm = ChatOpenAI(
            api_key=api_key, 
            model="gpt-3.5-turbo", 
            temperature=0.2
        )
        
    def optimize_itinerary(self, state: TourPlanningState) -> Dict[str, Any]:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Optimize the current itinerary considering budget, 
            travel time, and user preferences."""),
            ("human", "Current Itinerary: {itinerary}, Budget: {budget}")
        ])
        
        # Prepare the prompt with current itinerary and budget
        prompt = prompt_template.format_messages(
            itinerary=json.dumps(state.itinerary, indent=2),
            budget=state.user_preferences.get('budget', 0)
        )
        
        # Get the optimized itinerary from the LLM
        response = self.llm(prompt)
        
        # Update the itinerary with optimized content
        return {
            "itinerary": [{"optimized_description": response.content}],
            "current_step": "itinerary_optimized"
        }

# Agent to update the memory graph with user preferences
class MemoryAgent:
    def update_memory(self, state: TourPlanningState) -> Dict[str, Any]:
        memory_graph = {}
        for key, value in state.user_preferences.items():
            if key not in ['current_step', 'current_input', 'collected_preferences']:
                memory_graph[key] = value
        
        return {
            "memory_graph": memory_graph,
            "current_step": "memory_updated"
        }

# Agent to fetch weather recommendations based on the planned day
class WeatherAgent:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API Key must be set in .env file or environment variables")
        
        self.llm = ChatOpenAI(
            api_key=api_key, 
            model="gpt-3.5-turbo", 
            temperature=0.2
        )
    
    def get_weather_recommendations(self, state: TourPlanningState) -> Dict[str, Any]:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Provide weather recommendations for the planned day."""),
            ("human", "City: {city}, Date: {date}")
        ])
        
        # Prepare the prompt with city and date
        prompt = prompt_template.format_messages(
            city=state.user_preferences.get('city', ''),
            date=state.user_preferences.get('date', '')
        )
        
        # Get weather recommendations from the LLM
        response = self.llm(prompt)
        
        return {
            "weather_recommendations": response.content,
            "current_step": "weather_checked"
        }

# Function to create and compile the tour planning workflow graph
def create_tour_planning_graph():
    workflow = StateGraph(TourPlanningState)
    
    # Instantiate agents
    user_agent = UserInteractionAgent()
    itinerary_agent = ItineraryGenerationAgent()
    optimization_agent = OptimizationAgent()
    memory_agent = MemoryAgent()
    weather_agent = WeatherAgent()
    
    # Add nodes for each agent to the workflow
    workflow.add_node("user_interaction", user_agent.collect_preferences)
    workflow.add_node("itinerary_generation", itinerary_agent.generate_initial_itinerary)
    workflow.add_node("optimization", optimization_agent.optimize_itinerary)
    workflow.add_node("memory", memory_agent.update_memory)
    workflow.add_node("weather", weather_agent.get_weather_recommendations)
    
    # Define the workflow sequence
    workflow.set_entry_point("user_interaction")
    workflow.add_edge("user_interaction", "itinerary_generation")
    workflow.add_edge("itinerary_generation", "optimization")
    workflow.add_edge("optimization", "memory")
    workflow.add_edge("memory", "weather")
    workflow.add_edge("weather", END)
    
    return workflow.compile()

# Main function to run the tour planning assistant
def main():
    # Initialize the workflow graph
    tour_planning_graph = create_tour_planning_graph()
    
    # Define the initial state with user input and preferences
    initial_state = TourPlanningState(
        user_preferences={
            "current_input": "Hi, I'd like to plan a one-day trip in Rome.",
            "city": "Rome",
            "interests": ["historical sites", "food"],
            "budget": 150,
            "date": "2024-05-20"  # Ensure date is provided for WeatherAgent
        }
    )
    
    # Run the workflow graph with the initial state
    final_state = tour_planning_graph.invoke(initial_state)
    
    # Convert final state to a dictionary for easy access
    final_state_dict = final_state.to_dict() if hasattr(final_state, 'to_dict') else final_state
    
    # Print the final itinerary, memory graph, and weather recommendations
    print("Final Tour Plan:", json.dumps(final_state_dict.get('itinerary', []), indent=2))
    print("Memory Graph:", json.dumps(final_state_dict.get('memory_graph', {}), indent=2))
    print("Weather Recommendations:", final_state_dict.get('weather_recommendations', ""))

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
