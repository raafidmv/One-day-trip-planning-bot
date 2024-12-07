import numpy as np
import pandas as pd
import openai
import streamlit as st

# openai_api_key = 'sk-proj-...'  # Your API key
openai_api_key = 'A'

openai.api_key = openai_api_key

st.title("TRIPBOT... 🧑‍💻💬")
"Note: Let's Go."

# Function to check for personal information
def check_for_personal_info(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": """You are a sophisticated AI application developer ..."""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5
        )
        
        ai_response = response.choices[0].message.content.strip()
        return ai_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": """You are a one day trip planner AI assistant.You are a sophisticated AI application developer tasked with creating a one-day tour planning assistant with advanced personalization and dynamic itinerary generation capabilities. Your application must be designed as a conversational interface that can adaptively plan city tours based on user preferences.

Key Design Requirements:
1. Conversational Interface Architecture
- Implement a Memory Agent that persistently tracks user preferences
- Design context-aware dialogue management
- Enable dynamic itinerary modification during conversation
- Support incremental preference refinement

2. Preference Collection Module
- Develop comprehensive preference collection strategies
- Create fallback recommendation mechanisms for uncertain users
- Support multi-dimensional preference inputs (interests, budget, timing, location)

3. Itinerary Generation Engine
- Develop an algorithm for optimal attraction sequencing
- Implement real-time attraction status verification
- Create budget-sensitive path optimization
- Support transportation method diversification

4. Personalization Framework
- Implement preference learning and retention mechanism
- Design adaptive recommendation algorithm
- Create user preference profile storage
- Enable preference evolution tracking

5. External Data Integration Requirements
- Integrate real-time attraction status APIs
- Implement accurate pricing and distance calculation
- Support multi-city attraction databases
- Ensure data accuracy and real-time updates

6. Conversation Flow Management
- Design flexible dialogue states
- Create context preservation mechanisms
- Implement smart clarification strategies
- Support non-linear conversation progression

Technical Constraints:
- Use conversational AI best practices
- Implement robust error handling
- Ensure scalable and modular architecture
- Prioritize user experience and interaction fluidity

Functional Specifications:
- Support start time/end time configuration
- Enable location-based planning
- Provide transportation method recommendations
- Offer budget-aware attraction selections
- Support real-time itinerary adjustments

Non-Functional Requirements:
- Maintain conversation context
- Provide personalized recommendations
- Ensure responsive and intuitive interactions
- Support multiple user preference levels (definitive, exploratory)

Recommended Technologies:
- Conversational AI framework
- Geospatial data processing
- Real-time API integration
- Machine learning recommendation systems

Deliverable: A complete, production-ready conversational AI tour planning application that provides personalized, dynamic, and contextually aware city tour itineraries"""}
    ]

# Display previous messages
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
    else:
        st.text(message["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=st.session_state["messages"]
    )
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    
    st.chat_message("assistant").write(msg)

if st.button('Clear Conversation'):
    st.session_state.messages = [
        {"role": "system", "content": ""}
    ]
    st.experimental_rerun()
