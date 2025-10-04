# NIS Protocol JavaScript Client SDK

Simple, easy-to-use JavaScript/TypeScript client for interacting with NIS Protocol backend.

Works in both **Node.js** and **browsers**!

## Installation

### Browser

```html
<script src="nis-client.js"></script>
<script>
    const client = new NISClient('http://localhost:8000');
</script>
```

### Node.js

```bash
npm install node-fetch form-data  # For Node.js < 18
```

```javascript
const NISClient = require('./nis-client.js');
const client = new NISClient('http://localhost:8000');
```

## Quick Start

```javascript
const client = new NISClient('http://localhost:8000');

// Check health
const healthy = await client.isHealthy();
if (healthy) {
    console.log('✅ Backend is healthy!');
}

// Simple chat
const response = await client.chat('What is quantum computing?');
console.log('Response:', response.response);
console.log('Provider:', response.provider);
console.log('Tokens:', response.tokens_used);
```

## Features

### Chat Methods

```javascript
// Basic chat
const response = await client.chat('Hello!', {
    userId: 'user123',
    conversationId: 'conv456',
    provider: 'openai',  // openai, anthropic, google, deepseek, kimi, smart
    agentType: 'reasoning'  // reasoning, creative, analytical
});

// Smart Consensus (multiple LLMs)
const consensus = await client.smartConsensus('Explain machine learning');

// Streaming chat with real-time updates
await client.streamChat('Tell me a story', (data) => {
    if (data.type === 'content') {
        process.stdout.write(data.data);
    } else if (data.type === 'done') {
        console.log('\nDone!');
    } else if (data.type === 'error') {
        console.error('Error:', data.data);
    }
});
```

### Agent Methods

```javascript
// Get all agents
const agents = await client.getAgents();
agents.forEach(agent => {
    console.log(`${agent.name}: ${agent.status}`);
});

// Get specific agent
const agent = await client.getAgent('consciousness');
console.log(agent);
```

### Physics Methods

```javascript
// Validate physics scenario
const result = await client.validatePhysics(
    'Ball thrown at 45 degrees with initial velocity 20 m/s',
    {
        domain: 'mechanics',
        mode: 'true_pinn'
    }
);

// Get capabilities
const capabilities = await client.getPhysicsCapabilities();
console.log('Physics domains:', capabilities.domains);
```

### Research Methods

```javascript
// Deep research
const result = await client.deepResearch('Quantum computing applications', {
    depth: 'comprehensive',
    sources: 10
});

console.log('Research:', result.analysis);
console.log('Sources:', result.sources);
```

### Voice Methods (Browser)

```javascript
// Get voice settings
const settings = await client.getVoiceSettings();
console.log('Available speakers:', settings.available_speakers);

// Speech-to-Text
navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    const mediaRecorder = new MediaRecorder(stream);
    const chunks = [];
    
    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        const result = await client.transcribeAudio(audioBlob);
        console.log('Transcription:', result.text);
    };
    
    mediaRecorder.start();
    setTimeout(() => mediaRecorder.stop(), 5000);
});

// Text-to-Speech
const audioBlob = await client.synthesizeSpeech('Hello, world!', {
    speaker: 'consciousness',
    emotion: 'neutral',
    speed: 1.0
});

// Play audio
const audio = new Audio(URL.createObjectURL(audioBlob));
audio.play();
```

## Response Format

All responses follow this structure:

```javascript
{
    response: string,          // AI response text
    provider: string,          // LLM provider used
    model: string,            // Model name
    confidence: number,        // Response confidence (0-1)
    tokens_used: number,       // Tokens consumed
    real_ai: boolean,          // Real AI vs mock
    reasoning_trace: string[]  // Processing steps
}
```

## Error Handling

```javascript
try {
    const response = await client.chat('Hello!');
} catch (error) {
    console.error('Error:', error.message);
}
```

## Advanced Usage

### Custom API Key

```javascript
const client = new NISClient(
    'http://localhost:8000',
    'your-api-key'
);
```

### Custom Headers

The client automatically handles authentication headers when an API key is provided.

## Browser Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>NIS Protocol Client</title>
    <script src="nis-client.js"></script>
</head>
<body>
    <input id="messageInput" type="text" placeholder="Ask something...">
    <button onclick="sendMessage()">Send</button>
    <div id="response"></div>

    <script>
        const client = new NISClient('http://localhost:8000');

        async function sendMessage() {
            const message = document.getElementById('messageInput').value;
            const responseDiv = document.getElementById('response');
            
            try {
                const result = await client.chat(message);
                responseDiv.innerHTML = `<p>${result.response}</p>`;
            } catch (error) {
                responseDiv.innerHTML = `<p style="color:red">Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
```

## Node.js Example

```javascript
const NISClient = require('./nis-client.js');

async function main() {
    const client = new NISClient('http://localhost:8000');
    
    // Check health
    const healthy = await client.isHealthy();
    console.log('Backend healthy:', healthy);
    
    // Chat
    const response = await client.chat('What is AI?');
    console.log('Response:', response.response);
    
    // Get agents
    const agents = await client.getAgents();
    console.log('Active agents:', agents.length);
    
    // Physics
    const physics = await client.validatePhysics(
        'Projectile motion at 30 degrees',
        { domain: 'mechanics' }
    );
    console.log('Physics result:', physics);
}

main().catch(console.error);
```

## TypeScript Support

The client works with TypeScript. Create a declaration file:

```typescript
declare class NISClient {
    constructor(baseUrl?: string, apiKey?: string | null);
    
    chat(message: string, options?: {
        userId?: string;
        conversationId?: string;
        provider?: string;
        agentType?: string;
    }): Promise<any>;
    
    smartConsensus(message: string, options?: any): Promise<any>;
    streamChat(message: string, onMessage: (data: any) => void, options?: any): Promise<void>;
    
    getAgents(): Promise<any[]>;
    getAgent(agentName: string): Promise<any>;
    
    validatePhysics(scenario: string, options?: {
        domain?: string;
        mode?: string;
    }): Promise<any>;
    
    getPhysicsCapabilities(): Promise<any>;
    
    deepResearch(query: string, options?: {
        depth?: string;
        sources?: number;
    }): Promise<any>;
    
    getVoiceSettings(): Promise<any>;
    transcribeAudio(audioBlob: Blob): Promise<any>;
    synthesizeSpeech(text: string, options?: {
        speaker?: string;
        emotion?: string;
        speed?: number;
    }): Promise<Blob>;
    
    health(): Promise<any>;
    isHealthy(): Promise<boolean>;
    version(): Promise<string>;
}
```

## Requirements

### Browser
- Modern browser with Fetch API support
- ES6+ support

### Node.js
- Node.js 14+
- `node-fetch` (for Node < 18)
- `form-data` (for audio uploads)

## Examples

See integration examples:
- `examples/integrations/drone_integration.py`
- `examples/integrations/auto_integration.py`
- `examples/integrations/city_integration.py`

## License

Same as NIS Protocol main project

