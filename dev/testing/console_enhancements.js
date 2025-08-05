// ====== ENHANCED RESPONSE DISPLAY FUNCTIONS ======

function processResponse(data, metadata = {}) {
    // Determine response type and format accordingly
    console.log('Processing response:', data);
    
    if (data.formatted_content) {
        return formatFormattedResponse(data, metadata);
    } else if (data.generation) {
        return formatImageGenerationResponse(data, metadata);
    } else if (data.analysis) {
        return formatAnalysisResponse(data, metadata);
    } else if (data.research) {
        return formatResearchResponse(data, metadata);
    } else if (data.content || data.response) {
        return formatStandardResponse(data, metadata);
    } else {
        return formatRawResponse(data, metadata);
    }
}

function formatFormattedResponse(data, metadata) {
    const mode = data.output_mode || 'technical';
    const content = data.formatted_content || data.content;
    
    let html = `<div class="formatted-response ${mode}">`;
    
    // Add mode indicator
    html += `<div class="response-mode">
        <span class="mode-badge mode-${mode}">${mode.toUpperCase()}</span>
        <span class="audience-level">${data.audience_level || 'expert'}</span>
    </div>`;
    
    // Format content based on mode
    if (mode === 'visual') {
        html += formatVisualContent(content);
    } else if (mode === 'eli5') {
        html += formatELI5Content(content);
    } else if (mode === 'casual') {
        html += formatCasualContent(content);
    } else {
        html += formatTechnicalContent(content);
    }
    
    // Add confidence breakdown if available
    if (data.confidence_breakdown) {
        html += formatConfidenceBreakdown(data.confidence_breakdown);
    }
    
    // Add visual elements if available
    if (data.visual_elements) {
        html += formatVisualElements(data.visual_elements);
    }
    
    html += '</div>';
    return html;
}

function formatImageGenerationResponse(data, metadata) {
    const generation = data.generation;
    if (!generation) return formatRawResponse(data, metadata);
    
    let html = `<div class="image-generation-response">`;
    html += `<div class="generation-header">
        <h3>🎨 Image Generation Results</h3>
        <span class="provider-badge">${generation.provider_used || 'unknown'}</span>
    </div>`;
    
    if (generation.status === 'success' && generation.images && generation.images.length > 0) {
        html += `<div class="generation-success">`;
        
        generation.images.forEach((image, index) => {
            html += `<div class="generated-image">
                <div class="image-info">
                    <strong>Image ${index + 1}</strong>
                    <span class="image-size">${image.size || '1024x1024'}</span>
                </div>`;
            
            if (image.url && image.url.startsWith('data:image')) {
                html += `<img src="${image.url}" alt="Generated image" class="generated-image-display" />`;
            }
            
            if (image.revised_prompt) {
                html += `<div class="image-prompt">
                    <strong>Prompt:</strong> ${image.revised_prompt}
                </div>`;
            }
            
            html += `</div>`;
        });
        
        // Add generation info
        if (generation.generation_info) {
            html += `<div class="generation-details">
                <h4>Generation Details:</h4>
                <ul>
                    <li><strong>Model:</strong> ${generation.generation_info.model || 'unknown'}</li>
                    <li><strong>Real API:</strong> ${generation.generation_info.real_api ? '✅ Yes' : '❌ Mock'}</li>
                    <li><strong>Style:</strong> ${generation.style || 'default'}</li>
                    <li><strong>Quality:</strong> ${generation.quality || 'standard'}</li>
                </ul>
            </div>`;
        }
        
        html += `</div>`;
    } else {
        html += `<div class="generation-error">
            <p>❌ Image generation failed</p>
            <p><strong>Status:</strong> ${generation.status || 'unknown'}</p>
        </div>`;
    }
    
    html += '</div>';
    return html;
}

function formatAnalysisResponse(data, metadata) {
    const analysis = data.analysis;
    if (!analysis) return formatRawResponse(data, metadata);
    
    let html = `<div class="analysis-response">`;
    html += `<div class="analysis-header">
        <h3>🔍 Analysis Results</h3>
        <span class="confidence-score">Confidence: ${(analysis.confidence * 100).toFixed(1)}%</span>
    </div>`;
    
    if (analysis.description) {
        html += `<div class="analysis-description">
            <h4>Description:</h4>
            <p>${analysis.description}</p>
        </div>`;
    }
    
    if (analysis.objects && analysis.objects.length > 0) {
        html += `<div class="detected-objects">
            <h4>Detected Objects:</h4>
            <ul>`;
        analysis.objects.forEach(obj => {
            html += `<li>${obj.name} (${(obj.confidence * 100).toFixed(1)}%)</li>`;
        });
        html += `</ul></div>`;
    }
    
    if (analysis.tags && analysis.tags.length > 0) {
        html += `<div class="analysis-tags">
            <h4>Tags:</h4>
            <div class="tag-list">`;
        analysis.tags.forEach(tag => {
            html += `<span class="tag">${tag}</span>`;
        });
        html += `</div></div>`;
    }
    
    html += '</div>';
    return html;
}

function formatResearchResponse(data, metadata) {
    const research = data.research;
    if (!research) return formatRawResponse(data, metadata);
    
    let html = `<div class="research-response">`;
    html += `<div class="research-header">
        <h3>📚 Research Results</h3>
        <span class="sources-count">${research.sources?.length || 0} sources</span>
    </div>`;
    
    if (research.summary) {
        html += `<div class="research-summary">
            <h4>Summary:</h4>
            <p>${research.summary}</p>
        </div>`;
    }
    
    if (research.sources && research.sources.length > 0) {
        html += `<div class="research-sources">
            <h4>Sources:</h4>
            <ol>`;
        research.sources.forEach(source => {
            html += `<li>
                <a href="${source.url}" target="_blank">${source.title}</a>
                <span class="source-confidence">(Confidence: ${(source.confidence * 100).toFixed(1)}%)</span>
            </li>`;
        });
        html += `</ol></div>`;
    }
    
    html += '</div>';
    return html;
}

function formatVisualContent(content) {
    let html = `<div class="visual-content">`;
    
    // Handle content that includes images
    if (content.includes('🖼️') || content.includes('📊')) {
        html += `<div class="visual-indicator">
            <span class="visual-badge">VISUAL MODE</span>
        </div>`;
    }
    
    // Convert markdown-style content to HTML
    content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    content = content.replace(/\*(.+?)\*/g, '<em>$1</em>');
    content = content.replace(/🎨|📊|🖼️|💡/g, '<span class="visual-emoji">$&</span>');
    content = content.replace(/\n/g, '<br>');
    
    html += `<div class="visual-text">${content}</div>`;
    html += '</div>';
    return html;
}

function formatELI5Content(content) {
    let html = `<div class="eli5-content">`;
    html += `<div class="eli5-indicator">
        <span class="eli5-badge">ELI5 - Explain Like I'm 5</span>
        <span class="eli5-emoji">🧒</span>
    </div>`;
    
    // Make it more colorful and fun
    content = content.replace(/🌟|📚|🧠/g, '<span class="fun-emoji">$&</span>');
    content = content.replace(/\*\*(.+?)\*\*/g, '<span class="eli5-highlight">$1</span>');
    content = content.replace(/\n/g, '<br>');
    
    html += `<div class="eli5-text">${content}</div>`;
    html += '</div>';
    return html;
}

function formatCasualContent(content) {
    let html = `<div class="casual-content">`;
    html += `<div class="casual-indicator">
        <span class="casual-badge">CASUAL</span>
        <span class="casual-emoji">💬</span>
    </div>`;
    
    content = content.replace(/\n/g, '<br>');
    content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    
    html += `<div class="casual-text">${content}</div>`;
    html += '</div>';
    return html;
}

function formatTechnicalContent(content) {
    let html = `<div class="technical-content">`;
    html += `<div class="technical-indicator">
        <span class="technical-badge">TECHNICAL</span>
        <span class="technical-emoji">🔬</span>
    </div>`;
    
    content = content.replace(/\n/g, '<br>');
    content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    
    html += `<div class="technical-text">${content}</div>`;
    html += '</div>';
    return html;
}

function formatConfidenceBreakdown(breakdown) {
    let html = `<div class="confidence-breakdown">
        <h4>🎯 Confidence Breakdown</h4>
        <div class="confidence-metrics">`;
    
    for (const [key, value] of Object.entries(breakdown)) {
        if (typeof value === 'object' && value.value !== undefined) {
            const percentage = (value.value * 100).toFixed(1);
            html += `<div class="confidence-metric">
                <div class="metric-header">
                    <span class="metric-name">${key.replace('_', ' ').toUpperCase()}</span>
                    <span class="metric-value">${percentage}%</span>
                </div>
                <div class="metric-bar">
                    <div class="metric-fill" style="width: ${percentage}%"></div>
                </div>
                <div class="metric-explanation">${value.explanation}</div>
            </div>`;
        }
    }
    
    html += `</div></div>`;
    return html;
}

function formatStandardResponse(data, metadata) {
    const content = data.content || data.response || JSON.stringify(data);
    
    let html = `<div class="standard-response">`;
    
    if (typeof content === 'string') {
        html += `<div class="response-text">${content.replace(/\n/g, '<br>')}</div>`;
    } else {
        html += `<div class="response-json">
            <pre>${JSON.stringify(content, null, 2)}</pre>
        </div>`;
    }
    
    html += '</div>';
    return html;
}

function formatRawResponse(data, metadata) {
    let html = `<div class="raw-response">`;
    html += `<div class="raw-indicator">
        <span class="raw-badge">RAW DATA</span>
        <button onclick="this.parentElement.parentElement.querySelector('.raw-content').classList.toggle('expanded')">
            Toggle Details
        </button>
    </div>`;
    
    html += `<div class="raw-content">
        <pre>${JSON.stringify(data, null, 2)}</pre>
    </div>`;
    
    html += '</div>';
    return html;
}