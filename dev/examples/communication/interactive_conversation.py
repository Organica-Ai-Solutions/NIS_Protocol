"""
Interactive Conversation Example

This example demonstrates the InterpretationAgent and CommunicationAgent
working together to enable natural conversation with speech synthesis.
"""

import time
from typing import Dict, Any

from src.agents.interpretation.interpretation_agent import InterpretationAgent
from src.agents.communication.communication_agent import CommunicationAgent
from src.emotion.emotional_state import EmotionalState

def print_interpretation(interpretation: Dict[str, Any]) -> None:
    """Print interpretation results in a readable format."""
    print("\nInterpretation Results:")
    print("-" * 20)
    print(f"Sentiment: {interpretation.get('sentiment', 'Unknown')}")
    print("\nContent Types:")
    for content_type in interpretation.get('content_type', []):
        print(f"- {content_type['label']} ({content_type['score']:.2f})")
    print("\nKey Information:")
    for key, value in interpretation.get('key_information', {}).items():
        print(f"- {key}: {value}")
    print("-" * 20)

def main():
    # Initialize agents with emotional awareness
    emotional_state = EmotionalState()
    
    interpreter = InterpretationAgent(
        agent_id="interpreter",
        emotional_state=emotional_state,
        model_name="bert-base-uncased"
    )
    
    communicator = CommunicationAgent(
        agent_id="communicator",
        emotional_state=emotional_state,
        interpreter=interpreter,
        output_dir="data/audio_output"
    )
    
    print("\nWelcome to the Interactive Conversation Demo!")
    print("This demo shows how the NIS Protocol handles natural communication.")
    print("Type 'quit' to exit.")
    print("\nThe system will:")
    print("1. Interpret your input")
    print("2. Generate an appropriate response")
    print("3. Speak the response using Bark text-to-speech")
    print("\nStarting conversation...\n")
    
    # Initial greeting
    greeting = "Hello! I'm ready to chat with you. How are you feeling today?"
    communicator.process({
        "operation": "speak",
        "content": greeting
    })
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            farewell = "Thank you for chatting with me. Goodbye!"
            communicator.process({
                "operation": "speak",
                "content": farewell
            })
            break
        
        # Process user input
        print("\nProcessing...")
        
        # First interpret the input
        interpretation = interpreter.process({
            "operation": "interpret",
            "content": user_input
        })
        
        if interpretation["status"] == "success":
            print_interpretation(interpretation)
        
        # Generate and speak response
        response = communicator.process({
            "operation": "respond",
            "content": user_input
        })
        
        if response["status"] == "error":
            print(f"Error: {response['error']}")
            continue
        
        # Show conversation history
        print("\nConversation History:")
        print("-" * 20)
        for entry in communicator.get_conversation_history()[-5:]:
            role = entry["role"]
            content = entry["content"]
            print(f"{role.capitalize()}: {content}")
        print("-" * 20)
        
        # Small delay to prevent overwhelming
        time.sleep(1)

if __name__ == "__main__":
    main() 