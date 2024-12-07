import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TripPlannerBot:
    def __init__(self):
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.7, 
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        # OpenWeatherMap API Key
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')
        
        # Initialize prompt template for trip planning
        self.trip_prompt = PromptTemplate(
            input_variables=["location", "weather", "chat_history"],
            template="""
            You are an AI Trip Planner Assistant. 
            Location: {location}
            Current Weather: {weather}

            Based on the location and weather conditions, suggest a personalized one-day trip itinerary.
            Consider the following:
            1. Weather-appropriate activities
            2. Indoor/outdoor recommendations
            3. Local attractions
            4. Dining suggestions
            5. Transportation options

            Previous Conversation History: {chat_history}

            Provide a detailed, engaging itinerary that takes into account the current weather conditions.
            """
        )
        
        # Create trip planning chain
        self.trip_chain = LLMChain(
            llm=self.llm, 
            prompt=self.trip_prompt,
            memory=self.memory,
            verbose=True
        )
    
    def get_weather(self, location):
        """Fetch current weather for a given location"""
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': location,
            'appid': self.weather_api_key,
            'units': 'metric'  # Use metric for Celsius
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                weather_description = data['weather'][0]['description']
                temperature = data['main']['temp']
                return f"{weather_description.capitalize()}, Temperature: {temperature}°C"
            else:
                return "Unable to fetch weather information"
        
        except Exception as e:
            return f"Weather API Error: {str(e)}"
    
    def generate_follow_up_questions(self, location, weather):
        """Generate contextual follow-up questions"""
        follow_up_prompt = PromptTemplate(
            input_variables=["location", "weather"],
            template="""
            Generate 3-4 intelligent follow-up questions 
            that help refine a one-day trip plan for {location} 
            considering the current {weather} conditions.
            
            Questions should help understand:
            - User's interests
            - Preferred activity types
            - Budget considerations
            - Time constraints
            """
        )
        
        follow_up_chain = LLMChain(llm=self.llm, prompt=follow_up_prompt)
        return follow_up_chain.run(location=location, weather=weather)

def main():
    st.title("🌍 One-Day Trip Planner AI 🧳")
    
    # Initialize the bot
    bot = TripPlannerBot()
    
    # Session state for tracking conversation stage
    if 'stage' not in st.session_state:
        st.session_state.stage = 'initial'
    
    # Location input
    location = st.text_input("Enter your destination city", key="location_input")
    
    if location and st.button("Plan My Trip"):
        # Fetch weather
        weather_info = bot.get_weather(location)
        
        # Generate trip plan
        trip_plan = bot.trip_chain.run(
            location=location, 
            weather=weather_info
        )
        
        # Display trip plan
        st.markdown("### 🌞 Your Personalized Trip Plan")
        st.write(trip_plan)
        
        # Generate follow-up questions
        follow_up_questions = bot.generate_follow_up_questions(location, weather_info)
        
        st.markdown("### 🤔 Follow-up Questions")
        st.write(follow_up_questions)
        
        # Store in session state for context
        st.session_state.location = location
        st.session_state.weather = weather_info

    # Optional: Conversation continuation
    if 'location' in st.session_state:
        user_refinement = st.text_area("Refine your trip plan based on the follow-up questions")
        
        if user_refinement:
            # Use LangChain to process refinement
            refined_plan = bot.trip_chain.run(
                location=st.session_state.location, 
                weather=st.session_state.weather,
                chat_history=user_refinement
            )
            
            st.markdown("### 🔍 Refined Trip Plan")
            st.write(refined_plan)

if __name__ == "__main__":
    main()
