```mermaid
graph TD
    subgraph "Emotional State System"
    
    %% Emotional Dimensions
    subgraph "Emotional Dimensions"
        SUS["Suspicion
        (Decay: 0.05)"] 
        URG["Urgency
        (Decay: 0.10)"]
        CON["Confidence
        (Decay: 0.02)"]
        INT["Interest
        (Decay: 0.03)"]
        NOV["Novelty
        (Decay: 0.15)"]
    end
    
    %% Triggers
    subgraph "Input Triggers"
        UTG["Urgency Triggers
        - urgent
        - immediately
        - deadline
        - critical"]
        
        STG["Suspicion Triggers
        - unusual
        - suspicious
        - warning
        - alert"]
        
        NTG["Novelty Triggers
        - New patterns
        - Unknown inputs
        - Unexpected events"]
        
        CTG["Confidence Triggers
        - Confirmed predictions
        - Consistent patterns
        - Stable environment"]
        
        ITG["Interest Triggers
        - Relevant topics
        - High-value data
        - Priority domains"]
    end
    
    %% Effects
    subgraph "System Effects"
        URE["Higher task priority"]
        SUE["Increased scrutiny"]
        NOE["Focused attention"]
        COE["Lower decision threshold"]
        ITE["Increased resource allocation"]
    end
    
    %% Connections
    UTG --> URG
    STG --> SUS
    NTG --> NOV
    CTG --> CON
    ITG --> INT
    
    URG --> URE
    SUS --> SUE
    NOV --> NOE
    CON --> COE
    INT --> ITE
    
    %% Decay process
    DEC["Time-Based Decay
    - Each dimension decays toward 0.5 (neutral)
    - Different decay rates for each dimension
    - Fast decay: Novelty, Urgency
    - Moderate decay: Suspicion
    - Slow decay: Interest, Confidence"]
    
    URG -.-> DEC
    SUS -.-> DEC
    NOV -.-> DEC
    CON -.-> DEC
    INT -.-> DEC
    
    end
    
    %% Styling
    classDef dimension fill:#f9d4d4,stroke:#333,stroke-width:1px
    classDef trigger fill:#d4f9d4,stroke:#333,stroke-width:1px
    classDef effect fill:#d4d4f9,stroke:#333,stroke-width:1px
    classDef decay fill:#f9f9d4,stroke:#333,stroke-width:1px
    
    class SUS,URG,CON,INT,NOV dimension
    class UTG,STG,NTG,CTG,ITG trigger
    class URE,SUE,NOE,COE,ITE effect
    class DEC decay
```

# Emotional State System Diagram

This diagram illustrates the NIS Protocol's biologically inspired emotional state system, which modulates decision-making and resource allocation.

## Key Components

### Emotional Dimensions

The system maintains five core emotional dimensions, each with a different decay rate:

1. **Suspicion** - Increases scrutiny of unusual patterns (medium decay)
2. **Urgency** - Prioritizes time-sensitive processing (fast decay)
3. **Confidence** - Influences threshold for decision-making (slow decay)
4. **Interest** - Directs attention to specific features (slow decay)
5. **Novelty** - Highlights deviation from expectations (very fast decay)

### Input Triggers

Different stimuli trigger changes in emotional dimensions:
- Urgency keywords in text ("immediately", "deadline")
- Suspicious patterns in data
- Novel or unexpected inputs
- Confirmed predictions that build confidence
- Relevant data that increases interest

### System Effects

Emotional states directly influence system behavior:
- Heightened urgency increases task priority
- Higher suspicion triggers more detailed analysis
- Novelty directs focus toward new information
- Confidence adjusts decision thresholds
- Interest allocates computational resources

### Decay Process

The emotional state naturally decays toward a neutral state (0.5) over time:
- Each dimension has a specific decay rate
- Novelty decays the fastest (0.15 per time unit)
- Confidence decays the slowest (0.02 per time unit)
- Decay creates a natural "forgetting" curve

This emotional modulation system enables the NIS Protocol to prioritize information and actions in a human-like manner, balancing attention, caution, and resource allocation. 