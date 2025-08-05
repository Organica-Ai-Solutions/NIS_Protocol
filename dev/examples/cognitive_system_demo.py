from src.cognitive_agents.cognitive_system import CognitiveSystem
from src.cognitive_agents.reasoning_agent import ReasoningStrategy
import time

def print_separator():
    print("\n" + "="*80 + "\n")

def main():
    # Initialize the cognitive system
    system = CognitiveSystem()
    
    print("NIS Protocol Cognitive System Demo")
    print_separator()
    
    # Example 1: Basic sentiment and reasoning
    print("Example 1: Basic sentiment and reasoning")
    input_text = "I'm really excited about the potential of artificial intelligence to help solve complex problems in healthcare!"
    
    response = system.process_input(
        input_text,
        generate_speech=True  # This will generate and play speech
    )
    
    print(f"Input: {input_text}")
    print(f"Response:\n{response.response_text}")
    print_separator()
    
    # Example 2: Deductive reasoning
    print("Example 2: Deductive reasoning")
    input_text = "All birds have wings. A penguin is a bird. Therefore..."
    
    response = system.process_input(
        input_text,
        reasoning_strategy=ReasoningStrategy.DEDUCTIVE
    )
    
    print(f"Input: {input_text}")
    print(f"Response:\n{response.response_text}")
    print_separator()
    
    # Example 3: Causal reasoning with negative sentiment
    print("Example 3: Causal reasoning with negative sentiment")
    input_text = "The server crashed because of a memory leak, causing significant data loss and downtime."
    
    response = system.process_input(
        input_text,
        reasoning_strategy=ReasoningStrategy.CAUSAL
    )
    
    print(f"Input: {input_text}")
    print(f"Response:\n{response.response_text}")
    print_separator()
    
    # Example 4: Analogical reasoning
    print("Example 4: Analogical reasoning")
    input_text = "DNA|Computer Code"  # Format for analogical reasoning: source|target
    
    response = system.process_input(
        input_text,
        reasoning_strategy=ReasoningStrategy.ANALOGICAL
    )
    
    print(f"Input: Comparing {input_text.replace('|', ' to ')}")
    print(f"Response:\n{response.response_text}")
    print_separator()
    
    # Example 5: Conversation history
    print("Example 5: Conversation history")
    history = system.get_conversation_history()
    
    print("Recent conversation history:")
    for entry in history:
        print(f"\nSpeaker: {entry['speaker']}")
        print(f"Text: {entry['text']}")
        print(f"Timestamp: {entry['timestamp']}")
        if entry['sentiment']:
            print(f"Sentiment: {entry['sentiment']['label']} "
                  f"(score: {entry['sentiment']['score']:.2f})")
    
    # Clean up
    system.clear_state()

if __name__ == "__main__":
    main() 