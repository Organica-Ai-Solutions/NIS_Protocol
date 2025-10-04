/**
 * NIS Protocol JavaScript/TypeScript Client SDK
 * Simple, easy-to-use client for interacting with NIS Protocol backend
 * Works in Node.js and browsers
 */

class NISClient {
    /**
     * Create NIS Protocol client
     * @param {string} baseUrl - Backend URL (default: http://localhost:8000)
     * @param {string} apiKey - Optional API key for authentication
     */
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
    }

    /**
     * Make HTTP request to backend
     * @private
     */
    async _request(method, endpoint, data = null) {
        const url = `${this.baseUrl}${endpoint}`;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (this.apiKey) {
            options.headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    // ====== CHAT METHODS ======

    /**
     * Send chat message and get response
     * @param {string} message - User message
     * @param {Object} options - Chat options
     * @returns {Promise<Object>} Chat response
     */
    async chat(message, options = {}) {
        const data = {
            message,
            user_id: options.userId || 'default',
            conversation_id: options.conversationId || null,
            provider: options.provider || null,
            agent_type: options.agentType || 'reasoning'
        };

        return await this._request('POST', '/chat', data);
    }

    /**
     * Use Smart Consensus (multiple LLMs)
     * @param {string} message - User message
     * @param {Object} options - Chat options
     * @returns {Promise<Object>} Consensus response
     */
    async smartConsensus(message, options = {}) {
        return await this.chat(message, { ...options, provider: 'smart' });
    }

    /**
     * Stream chat response (SSE)
     * @param {string} message - User message
     * @param {Function} onMessage - Callback for each message chunk
     * @param {Object} options - Chat options
     */
    async streamChat(message, onMessage, options = {}) {
        const data = {
            message,
            user_id: options.userId || 'default',
            conversation_id: options.conversationId || null,
            provider: options.provider || null,
            agent_type: options.agentType || 'reasoning'
        };

        const url = `${this.baseUrl}/chat/stream`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(this.apiKey ? { 'Authorization': `Bearer ${this.apiKey}` } : {})
            },
            body: JSON.stringify(data)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    onMessage(data);
                }
            }
        }
    }

    // ====== AGENT METHODS ======

    /**
     * Get all registered agents
     * @returns {Promise<Array>} List of agents
     */
    async getAgents() {
        const result = await this._request('GET', '/agents/status');
        return result.agents || [];
    }

    /**
     * Get specific agent status
     * @param {string} agentName - Agent name
     * @returns {Promise<Object>} Agent status
     */
    async getAgent(agentName) {
        const agents = await this.getAgents();
        return agents.find(a => a.name === agentName);
    }

    // ====== PHYSICS METHODS ======

    /**
     * Validate physics scenario
     * @param {string} scenario - Physics scenario description
     * @param {Object} options - Validation options
     * @returns {Promise<Object>} Physics validation results
     */
    async validatePhysics(scenario, options = {}) {
        const data = {
            scenario,
            domain: options.domain || 'mechanics',
            mode: options.mode || 'true_pinn'
        };

        return await this._request('POST', '/physics/validate', data);
    }

    /**
     * Get physics system capabilities
     * @returns {Promise<Object>} Physics capabilities
     */
    async getPhysicsCapabilities() {
        return await this._request('GET', '/physics/capabilities');
    }

    // ====== RESEARCH METHODS ======

    /**
     * Perform deep research
     * @param {string} query - Research query
     * @param {Object} options - Research options
     * @returns {Promise<Object>} Research results
     */
    async deepResearch(query, options = {}) {
        const data = {
            query,
            depth: options.depth || 'comprehensive',
            max_sources: options.sources || 10
        };

        return await this._request('POST', '/research/deep', data);
    }

    // ====== VOICE METHODS ======

    /**
     * Get voice settings and capabilities
     * @returns {Promise<Object>} Voice settings
     */
    async getVoiceSettings() {
        return await this._request('GET', '/voice/settings');
    }

    /**
     * Transcribe audio (Speech-to-Text)
     * @param {Blob|File} audioBlob - Audio data
     * @returns {Promise<Object>} Transcription result
     */
    async transcribeAudio(audioBlob) {
        const formData = new FormData();
        formData.append('audio_data', audioBlob);

        const url = `${this.baseUrl}/communication/transcribe`;
        const response = await fetch(url, {
            method: 'POST',
            headers: this.apiKey ? { 'Authorization': `Bearer ${this.apiKey}` } : {},
            body: formData
        });

        return await response.json();
    }

    /**
     * Synthesize speech (Text-to-Speech)
     * @param {string} text - Text to synthesize
     * @param {Object} options - Synthesis options
     * @returns {Promise<Blob>} Audio blob
     */
    async synthesizeSpeech(text, options = {}) {
        const data = {
            text,
            speaker: options.speaker || 'consciousness',
            emotion: options.emotion || 'neutral',
            speed: options.speed || 1.0
        };

        const url = `${this.baseUrl}/communication/synthesize`;
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(this.apiKey ? { 'Authorization': `Bearer ${this.apiKey}` } : {})
            },
            body: JSON.stringify(data)
        });

        return await response.blob();
    }

    // ====== UTILITY METHODS ======

    /**
     * Check backend health
     * @returns {Promise<Object>} Health status
     */
    async health() {
        return await this._request('GET', '/health');
    }

    /**
     * Check if backend is healthy
     * @returns {Promise<boolean>} True if healthy
     */
    async isHealthy() {
        try {
            const health = await this.health();
            return health.status === 'healthy';
        } catch {
            return false;
        }
    }

    /**
     * Get backend version
     * @returns {Promise<string>} Version string
     */
    async version() {
        const health = await this.health();
        return health.pattern || 'unknown';
    }
}

// ====== BROWSER & NODE.JS COMPATIBILITY ======

if (typeof module !== 'undefined' && module.exports) {
    // Node.js
    module.exports = NISClient;
    
    // Add fetch polyfill for Node.js < 18
    if (typeof fetch === 'undefined') {
        global.fetch = require('node-fetch');
        global.FormData = require('form-data');
    }
}

if (typeof window !== 'undefined') {
    // Browser
    window.NISClient = NISClient;
}

// ====== EXAMPLE USAGE ======

/*
// Browser example:
const client = new NISClient('http://localhost:8000');

// Simple chat
client.chat('What is quantum computing?').then(response => {
    console.log('Response:', response.response);
    console.log('Provider:', response.provider);
    console.log('Tokens:', response.tokens_used);
});

// Smart Consensus
client.smartConsensus('Explain machine learning').then(response => {
    console.log('Consensus:', response.response);
});

// Streaming chat
client.streamChat('Tell me a story', (data) => {
    if (data.type === 'content') {
        process.stdout.write(data.data);
    } else if (data.type === 'done') {
        console.log('\nDone!');
    }
});

// Get agents
client.getAgents().then(agents => {
    console.log('Active agents:', agents.length);
    agents.forEach(agent => {
        console.log(`- ${agent.name} (${agent.type})`);
    });
});

// Physics validation
client.validatePhysics(
    'Ball thrown at 45 degrees with initial velocity 20 m/s',
    { domain: 'mechanics' }
).then(result => {
    console.log('Physics result:', result);
});

// Deep research
client.deepResearch('Quantum computing applications').then(research => {
    console.log('Research:', research);
});

// Voice (browser only)
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
*/

