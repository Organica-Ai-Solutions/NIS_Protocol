/**
 * Chat Performance Optimization Module
 * Improves async rendering and reduces DOM operations
 */

// Optimized message rendering with batching
class MessageRenderer {
    constructor() {
        this.pendingUpdates = new Map();
        this.rafId = null;
        this.updateQueue = [];
    }

    // Batch DOM updates using requestAnimationFrame
    scheduleUpdate(elementId, content, usePlainText = false) {
        this.updateQueue.push({ elementId, content, usePlainText });
        
        if (!this.rafId) {
            this.rafId = requestAnimationFrame(() => {
                this.flushUpdates();
            });
        }
    }

    flushUpdates() {
        // Process all pending updates in a single batch
        const updates = this.updateQueue.splice(0);
        
        updates.forEach(({ elementId, content, usePlainText }) => {
            const element = document.getElementById(elementId);
            if (element) {
                if (usePlainText) {
                    element.textContent = content;
                } else {
                    element.innerHTML = content;
                }
            }
        });

        this.rafId = null;
    }

    // Optimized append (for streaming)
    appendContent(element, content) {
        if (!element) return;
        
        // Use DocumentFragment for better performance
        const fragment = document.createDocumentFragment();
        const span = document.createElement('span');
        span.textContent = content;
        fragment.appendChild(span);
        element.appendChild(fragment);
    }
}

// Optimized scroll manager
class ScrollManager {
    constructor() {
        this.isScrolling = false;
        this.scrollTarget = null;
    }

    smoothScroll(element) {
        if (this.isScrolling) return;
        
        this.isScrolling = true;
        this.scrollTarget = element;

        requestAnimationFrame(() => {
            if (this.scrollTarget) {
                this.scrollTarget.scrollTop = this.scrollTarget.scrollHeight;
            }
            this.isScrolling = false;
        });
    }

    // Throttled scroll for streaming
    throttledScroll(element, delay = 16) {
        if (!this.scrollTimeout) {
            this.scrollTimeout = setTimeout(() => {
                this.smoothScroll(element);
                this.scrollTimeout = null;
            }, delay);
        }
    }
}

// Streaming response handler with optimization
class StreamingResponseHandler {
    constructor(contentElement, scrollElement) {
        this.contentElement = contentElement;
        this.scrollElement = scrollElement;
        this.buffer = '';
        this.updateInterval = null;
        this.chunkSize = 5; // Update every N chunks
        this.chunkCount = 0;
        this.scrollManager = new ScrollManager();
    }

    async processStream(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let metadata = {};

        try {
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim() !== '');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.type === 'content') {
                                fullResponse += data.data;
                                
                                // Batch updates for better performance
                                this.chunkCount++;
                                if (this.chunkCount >= this.chunkSize) {
                                    this.contentElement.textContent = fullResponse;
                                    this.scrollManager.throttledScroll(this.scrollElement);
                                    this.chunkCount = 0;
                                }
                            } else if (data.type === 'metadata') {
                                Object.assign(metadata, data.data);
                            }
                        } catch (e) {
                            console.warn('Stream parse error:', e);
                        }
                    }
                }
            }

            // Final update
            this.contentElement.textContent = fullResponse;
            this.scrollManager.smoothScroll(this.scrollElement);

            return { fullResponse, metadata };
        } catch (error) {
            console.error('Stream processing error:', error);
            throw error;
        }
    }

    cleanup() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Export for use in chat interfaces
window.ChatOptimization = {
    MessageRenderer,
    ScrollManager,
    StreamingResponseHandler
};

