/**
 * NIS Protocol State Management Client
 * Real-time WebSocket client for frontend state synchronization
 * 
 * This client automatically connects to the NIS backend and receives
 * real-time state updates.
 */

class NISStateClient {
    constructor(options = {}) {
        this.options = {
            baseUrl: options.baseUrl || `ws://${window.location.host}`,
            connectionType: options.connectionType || 'dashboard',
            userId: options.userId || null,
            sessionId: options.sessionId || `session_${Date.now()}`,
            autoReconnect: options.autoReconnect !== false,
            reconnectInterval: options.reconnectInterval || 5000,
            maxReconnectAttempts: options.maxReconnectAttempts || 10,
            debug: options.debug || false,
            ...options
        };
        
        this.websocket = null;
        this.connectionId = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        
        // State management
        this.currentState = {};
        this.eventListeners = new Map();
        this.stateSubscribers = new Map();
        
        // Performance metrics
        this.metrics = {
            messagesReceived: 0,
            messagesSent: 0,
            connectionTime: null,
            lastPing: null,
            averageLatency: 0
        };
        
        this.log('🧠 NIS State Client initialized', this.options);
        
        // Auto-connect if enabled
        if (options.autoConnect !== false) {
            this.connect();
        }
    }
    
    /**
     * Connect to NIS Protocol WebSocket
     */
    async connect() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.log('Already connected');
            return;
        }
        
        try {
            const params = new URLSearchParams();
            if (this.options.userId) params.append('user_id', this.options.userId);
            if (this.options.sessionId) params.append('session_id', this.options.sessionId);
            
            const wsUrl = `${this.options.baseUrl}/ws/state/${this.options.connectionType}?${params}`;
            
            this.log('🔌 Connecting to:', wsUrl);
            
            this.websocket = new WebSocket(wsUrl);
            this.setupWebSocketHandlers();
            
        } catch (error) {
            this.log('❌ Connection failed:', error);
            this.handleConnectionError(error);
        }
    }
    
    /**
     * Setup WebSocket event handlers
     */
    setupWebSocketHandlers() {
        this.websocket.onopen = (event) => {
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.metrics.connectionTime = Date.now();
            
            this.log('✅ Connected to NIS Protocol backend');
            
            // Start ping interval
            this.startPingInterval();
            
            // Emit connection event
            this.emit('connected', { event });
        };
        
        this.websocket.onmessage = (event) => {
            this.metrics.messagesReceived++;
            
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (error) {
                this.log('❌ Failed to parse message:', error, event.data);
            }
        };
        
        this.websocket.onclose = (event) => {
            this.isConnected = false;
            this.connectionId = null;
            
            this.log('🔌 Connection closed:', event.code, event.reason);
            
            // Stop ping interval
            this.stopPingInterval();
            
            // Emit disconnection event
            this.emit('disconnected', { event });
            
            // Auto-reconnect if enabled
            if (this.options.autoReconnect && this.reconnectAttempts < this.options.maxReconnectAttempts) {
                this.scheduleReconnect();
            }
        };
        
        this.websocket.onerror = (error) => {
            this.log('❌ WebSocket error:', error);
            this.emit('error', { error });
        };
    }
    
    /**
     * Handle incoming messages from backend
     */
    handleMessage(message) {
        this.log('📨 Received:', message);
        
        const { type, data, event_type, timestamp } = message;
        
        switch (type) {
            case 'initial_state':
                this.handleInitialState(data);
                break;
                
            case 'pong':
                this.handlePong(timestamp);
                break;
                
            case 'subscription_confirmed':
            case 'unsubscription_confirmed':
                this.log(`✅ ${type}:`, message);
                break;
                
            default:
                // Handle state events
                if (event_type) {
                    this.handleStateEvent(message);
                } else {
                    this.log('🤔 Unknown message type:', type);
                }
        }
    }
    
    /**
     * Handle initial state from backend
     */
    handleInitialState(state) {
        this.currentState = { ...state };
        this.log('📊 Initial state received:', Object.keys(state));
        
        // Notify all state subscribers
        this.notifyStateSubscribers('initial_state', state);
        
        // Emit initial state event
        this.emit('initial_state', { state });
    }
    
    /**
     * Handle state events from backend
     */
    handleStateEvent(event) {
        const { event_type, data, event_id, timestamp } = event;
        
        // Update local state if needed
        if (event_type === 'ui_state_update' && data.new_state) {
            Object.assign(this.currentState, data.new_state);
            this.notifyStateSubscribers('state_update', data.new_state);
        }
        
        // Emit specific event
        this.emit(event_type, { data, event_id, timestamp });
        
        // Emit generic state event
        this.emit('state_event', event);
    }
    
    /**
     * Handle pong response
     */
    handlePong(timestamp) {
        if (this.metrics.lastPing) {
            const latency = Date.now() - this.metrics.lastPing;
            this.metrics.averageLatency = (this.metrics.averageLatency + latency) / 2;
        }
        this.log('🏓 Pong received, latency:', this.metrics.averageLatency.toFixed(2), 'ms');
    }
    
    /**
     * Send message to backend
     */
    send(message) {
        if (!this.isConnected) {
            this.log('❌ Cannot send message: not connected');
            return false;
        }
        
        try {
            this.websocket.send(JSON.stringify(message));
            this.metrics.messagesSent++;
            this.log('📤 Sent:', message);
            return true;
        } catch (error) {
            this.log('❌ Failed to send message:', error);
            return false;
        }
    }
    
    /**
     * Subscribe to specific events
     */
    subscribeToEvents(events) {
        return this.send({
            type: 'subscribe',
            events: Array.isArray(events) ? events : [events]
        });
    }
    
    /**
     * Unsubscribe from events
     */
    unsubscribeFromEvents(events) {
        return this.send({
            type: 'unsubscribe',
            events: Array.isArray(events) ? events : [events]
        });
    }
    
    /**
     * Request current state from backend
     */
    requestState() {
        return this.send({ type: 'request_state' });
    }
    
    /**
     * Start ping interval for connection health
     */
    startPingInterval() {
        this.pingInterval = setInterval(() => {
            if (this.isConnected) {
                this.metrics.lastPing = Date.now();
                this.send({ type: 'ping', timestamp: this.metrics.lastPing });
            }
        }, 30000); // Ping every 30 seconds
    }
    
    /**
     * Stop ping interval
     */
    stopPingInterval() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }
        
        this.reconnectAttempts++;
        const delay = Math.min(this.options.reconnectInterval * this.reconnectAttempts, 30000);
        
        this.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.options.maxReconnectAttempts})`);
        
        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, delay);
    }
    
    /**
     * Handle connection errors
     */
    handleConnectionError(error) {
        this.emit('connection_error', { error });
        
        if (this.options.autoReconnect && this.reconnectAttempts < this.options.maxReconnectAttempts) {
            this.scheduleReconnect();
        }
    }
    
    /**
     * Add event listener
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event).add(callback);
        
        return () => this.off(event, callback);
    }
    
    /**
     * Remove event listener
     */
    off(event, callback) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).delete(callback);
        }
    }
    
    /**
     * Emit event to listeners
     */
    emit(event, data) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    this.log('❌ Event listener error:', error);
                }
            });
        }
    }
    
    /**
     * Subscribe to state changes
     */
    onStateChange(callback) {
        const id = `state_${Date.now()}_${Math.random()}`;
        this.stateSubscribers.set(id, callback);
        return () => this.stateSubscribers.delete(id);
    }
    
    /**
     * Notify state subscribers
     */
    notifyStateSubscribers(type, data) {
        this.stateSubscribers.forEach(callback => {
            try {
                callback(type, data, this.currentState);
            } catch (error) {
                this.log('❌ State subscriber error:', error);
            }
        });
    }
    
    /**
     * Get current state
     */
    getState() {
        return { ...this.currentState };
    }
    
    /**
     * Get connection metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            isConnected: this.isConnected,
            connectionId: this.connectionId,
            reconnectAttempts: this.reconnectAttempts
        };
    }
    
    /**
     * Disconnect from backend
     */
    disconnect() {
        this.options.autoReconnect = false;
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        this.stopPingInterval();
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.log('🔌 Disconnected from NIS Protocol backend');
    }
    
    /**
     * Debug logging
     */
    log(...args) {
        if (this.options.debug) {
            console.log('[NIS State Client]', ...args);
        }
    }
}

/**
 * NIS State Manager - Higher level state management
 */
class NISStateManager {
    constructor(options = {}) {
        this.client = new NISStateClient({
            debug: true,
            ...options
        });
        
        this.state = {};
        this.components = new Map();
        
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        // Handle initial state
        this.client.on('initial_state', ({ state }) => {
            this.state = { ...state };
            this.updateAllComponents();
        });
        
        // Handle state updates
        this.client.on('state_event', (event) => {
            this.handleStateEvent(event);
        });
        
        // Handle specific events
        this.client.on('system_status_change', ({ data }) => {
            this.updateSystemStatus(data);
        });
        
        this.client.on('agent_status_change', ({ data }) => {
            this.updateAgentStatus(data);
        });
        
        this.client.on('chat_message', ({ data }) => {
            this.handleChatMessage(data);
        });
    }
    
    /**
     * Register a component for automatic updates
     */
    registerComponent(id, component) {
        this.components.set(id, component);
        
        // Send current state to component
        if (component.onStateUpdate && Object.keys(this.state).length > 0) {
            component.onStateUpdate(this.state);
        }
        
        return () => this.components.delete(id);
    }
    
    /**
     * Update all registered components
     */
    updateAllComponents() {
        this.components.forEach((component, id) => {
            try {
                if (component.onStateUpdate) {
                    component.onStateUpdate(this.state);
                }
            } catch (error) {
                console.error(`Component ${id} update failed:`, error);
            }
        });
    }
    
    /**
     * Handle state events
     */
    handleStateEvent(event) {
        const { event_type, data } = event;
        
        // Update local state
        if (event_type === 'ui_state_update' && data.new_state) {
            Object.assign(this.state, data.new_state);
            this.updateAllComponents();
        }
        
        // Notify specific components
        this.components.forEach((component, id) => {
            try {
                if (component.onEvent) {
                    component.onEvent(event_type, data);
                }
            } catch (error) {
                console.error(`Component ${id} event handler failed:`, error);
            }
        });
    }
    
    /**
     * Update system status display
     */
    updateSystemStatus(data) {
        // Update system status indicators
        const statusElements = document.querySelectorAll('[data-nis-system-status]');
        statusElements.forEach(el => {
            if (data.system_health) {
                el.textContent = data.system_health;
                el.className = `status ${data.system_health}`;
            }
        });
    }
    
    /**
     * Update agent status display
     */
    updateAgentStatus(data) {
        // Update agent count displays
        const agentElements = document.querySelectorAll('[data-nis-agent-count]');
        agentElements.forEach(el => {
            if (data.active_agents) {
                el.textContent = Object.keys(data.active_agents).length;
            }
        });
    }
    
    /**
     * Handle chat messages
     */
    handleChatMessage(data) {
        // Emit custom event for chat components
        window.dispatchEvent(new CustomEvent('nis-chat-message', { detail: data }));
    }
    
    /**
     * Get current state
     */
    getState() {
        return { ...this.state };
    }
    
    /**
     * Get client metrics
     */
    getMetrics() {
        return this.client.getMetrics();
    }
}

// Global instance for easy access
window.NISStateClient = NISStateClient;
window.NISStateManager = NISStateManager;

// Auto-initialize if in browser environment
if (typeof window !== 'undefined') {
    // Wait for DOM to be ready before initializing
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            if (!window.nisStateManager) {
                window.nisStateManager = new NISStateManager({
                    debug: true,
                    autoConnect: true
                });

                console.log('🧠 NIS State Manager auto-initialized after DOM ready');
            }
        });
    } else {
        // DOM is already ready
        if (!window.nisStateManager) {
            window.nisStateManager = new NISStateManager({
                debug: true,
                autoConnect: true
            });

            console.log('🧠 NIS State Manager auto-initialized (DOM already ready)');
        }
    }
}
