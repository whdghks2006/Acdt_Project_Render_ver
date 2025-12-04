/**
 * Template Manager for AI Smart Scheduler
 * Manages quick input templates using localStorage
 */

class TemplateManager {
    constructor() {
        this.storageKey = 'ai_scheduler_templates';
        this.templates = this.loadTemplates();
    }

    /**
     * Load templates from localStorage
     */
    loadTemplates() {
        try {
            const data = localStorage.getItem(this.storageKey);
            return data ? JSON.parse(data) : [];
        } catch (error) {
            console.error('Error loading templates:', error);
            return [];
        }
    }

    /**
     * Save templates to localStorage
     */
    saveTemplates() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.templates));
            return true;
        } catch (error) {
            console.error('Error saving templates:', error);
            return false;
        }
    }

    /**
     * Create a new template from current form data
     */
    createTemplate(name, formData) {
        const template = {
            id: 'template_' + Date.now(),
            name: name,
            data: formData,
            created: new Date().toISOString()
        };

        this.templates.push(template);
        this.saveTemplates();
        return template;
    }

    /**
     * Get all templates
     */
    getAllTemplates() {
        return this.templates;
    }

    /**
     * Get template by ID
     */
    getTemplateById(id) {
        return this.templates.find(t => t.id === id);
    }

    /**
     * Update template name
     */
    updateTemplateName(id, newName) {
        const template = this.getTemplateById(id);
        if (template) {
            template.name = newName;
            this.saveTemplates();
            return true;
        }
        return false;
    }

    /**
     * Delete template by ID
     */
    deleteTemplate(id) {
        const index = this.templates.findIndex(t => t.id === id);
        if (index !== -1) {
            this.templates.splice(index, 1);
            this.saveTemplates();
            return true;
        }
        return false;
    }

    /**
     * Get current form data
     */
    getCurrentFormData() {
        return {
            summary: document.getElementById('summary-box')?.value || '',
            start_date: document.getElementById('start-date-box')?.value || '',
            end_date: document.getElementById('end-date-box')?.value || '',
            start_time: document.getElementById('start-time-box')?.value || '',
            end_time: document.getElementById('end-time-box')?.value || '',
            location: document.getElementById('loc-box')?.value || '',
            description: document.getElementById('description-box')?.value || '',
            allday: document.getElementById('allday-check')?.checked || false
        };
    }

    /**
     * Apply template data to form
     */
    applyTemplateToForm(templateId) {
        const template = this.getTemplateById(templateId);
        if (!template) return false;

        const data = template.data;

        // Fill form fields
        if (document.getElementById('summary-box'))
            document.getElementById('summary-box').value = data.summary || '';
        if (document.getElementById('start-date-box'))
            document.getElementById('start-date-box').value = data.start_date || '';
        if (document.getElementById('end-date-box'))
            document.getElementById('end-date-box').value = data.end_date || '';
        if (document.getElementById('start-time-box'))
            document.getElementById('start-time-box').value = data.start_time || '';
        if (document.getElementById('end-time-box'))
            document.getElementById('end-time-box').value = data.end_time || '';
        if (document.getElementById('loc-box')) {
            document.getElementById('loc-box').value = data.location || '';
            // Trigger map links update
            if (typeof updateMapLinks === 'function') {
                updateMapLinks();
            }
        }
        if (document.getElementById('description-box'))
            document.getElementById('description-box').value = data.description || '';
        if (document.getElementById('allday-check'))
            document.getElementById('allday-check').checked = data.allday || false;

        return true;
    }
}

// Global template manager instance
let templateManager;

// Initialize template manager when DOM is ready
document.addEventListener('DOMContentLoaded', function () {
    templateManager = new TemplateManager();
    updateTemplateDropdown();
});

/**
 * Update template dropdown with current templates
 */
function updateTemplateDropdown() {
    if (!templateManager) return;

    const dropdown = document.getElementById('template-select');
    if (!dropdown) return;

    const templates = templateManager.getAllTemplates();

    // Clear existing options except the first one (placeholder)
    dropdown.innerHTML = '<option value="" disabled selected id="opt-select-template">Select a template...</option>';

    // Add template options
    templates.forEach(template => {
        const option = document.createElement('option');
        option.value = template.id;
        option.textContent = template.name;
        dropdown.appendChild(option);
    });

    // Update count badge
    const countBadge = document.getElementById('template-count');
    if (countBadge) {
        countBadge.textContent = templates.length;
        countBadge.style.display = templates.length > 0 ? 'inline-block' : 'none';
    }
}

/**
 * Handle template selection from dropdown
 */
function handleTemplateSelect() {
    const dropdown = document.getElementById('template-select');
    if (!dropdown || !dropdown.value) return;

    const success = templateManager.applyTemplateToForm(dropdown.value);

    if (success) {
        // Show success message
        showTemplateNotification('Template applied successfully! ✓', 'success');

        // Reset dropdown
        dropdown.value = '';
    } else {
        showTemplateNotification('Failed to apply template', 'error');
    }
}

/**
 * Save current form as template
 */
function saveAsTemplate() {
    if (!templateManager) return;

    // Get current form data
    const formData = templateManager.getCurrentFormData();

    // Check if there's any data to save
    if (!formData.summary && !formData.start_date && !formData.location) {
        showTemplateNotification('Please fill in some event details first', 'warning');
        return;
    }

    // Prompt for template name
    const templateName = prompt(getLocalizedText('prompt-template-name', 'Enter template name:'));

    if (!templateName || templateName.trim() === '') {
        return; // User cancelled or entered empty name
    }

    // Create template
    const template = templateManager.createTemplate(templateName.trim(), formData);

    if (template) {
        updateTemplateDropdown();
        showTemplateNotification(`Template "${templateName}" saved! 💾`, 'success');
    } else {
        showTemplateNotification('Failed to save template', 'error');
    }
}

/**
 * Open template management modal
 */
function openTemplateManager() {
    const modal = document.getElementById('template-modal');
    if (!modal) return;

    // Update template list
    updateTemplateList();

    // Show modal
    modal.style.display = 'flex';
}

/**
 * Close template management modal
 */
function closeTemplateManager() {
    const modal = document.getElementById('template-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

/**
 * Update template list in management modal
 */
function updateTemplateList() {
    if (!templateManager) return;

    const listContainer = document.getElementById('template-list');
    if (!listContainer) return;

    const templates = templateManager.getAllTemplates();

    if (templates.length === 0) {
        listContainer.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #94a3b8;">
                <div style="font-size: 3rem; margin-bottom: 10px;">📋</div>
                <div id="txt-no-templates">No templates saved yet</div>
            </div>
        `;
        return;
    }

    listContainer.innerHTML = templates.map(template => `
        <div class="template-item" data-id="${template.id}">
            <div class="template-info">
                <div class="template-name">${escapeHtml(template.name)}</div>
                <div class="template-meta">
                    ${template.data.summary ? `📅 ${escapeHtml(template.data.summary)}` : ''}
                    ${template.data.location ? ` • 📍 ${escapeHtml(template.data.location)}` : ''}
                </div>
                <div class="template-date">Created: ${new Date(template.created).toLocaleDateString()}</div>
            </div>
            <div class="template-actions">
                <button onclick="applyTemplateFromModal('${template.id}')" class="btn-template-action btn-apply" title="Apply">
                    ✓
                </button>
                <button onclick="renameTemplate('${template.id}')" class="btn-template-action btn-rename" title="Rename">
                    ✏️
                </button>
                <button onclick="deleteTemplateConfirm('${template.id}')" class="btn-template-action btn-delete" title="Delete">
                    🗑️
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * Apply template from modal
 */
function applyTemplateFromModal(templateId) {
    const success = templateManager.applyTemplateToForm(templateId);

    if (success) {
        closeTemplateManager();
        showTemplateNotification('Template applied successfully! ✓', 'success');
    } else {
        showTemplateNotification('Failed to apply template', 'error');
    }
}

/**
 * Rename template
 */
function renameTemplate(templateId) {
    const template = templateManager.getTemplateById(templateId);
    if (!template) return;

    const newName = prompt(getLocalizedText('prompt-rename-template', 'Enter new name:'), template.name);

    if (newName && newName.trim() !== '' && newName !== template.name) {
        if (templateManager.updateTemplateName(templateId, newName.trim())) {
            updateTemplateList();
            updateTemplateDropdown();
            showTemplateNotification('Template renamed successfully', 'success');
        } else {
            showTemplateNotification('Failed to rename template', 'error');
        }
    }
}

/**
 * Delete template with confirmation
 */
function deleteTemplateConfirm(templateId) {
    const template = templateManager.getTemplateById(templateId);
    if (!template) return;

    const confirmMsg = getLocalizedText('confirm-delete-template',
        `Are you sure you want to delete "${template.name}"?`);

    if (confirm(confirmMsg)) {
        if (templateManager.deleteTemplate(templateId)) {
            updateTemplateList();
            updateTemplateDropdown();
            showTemplateNotification('Template deleted', 'success');
        } else {
            showTemplateNotification('Failed to delete template', 'error');
        }
    }
}

/**
 * Show template notification
 */
function showTemplateNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `template-toast template-toast-${type}`;
    notification.textContent = message;

    // Add to body
    document.body.appendChild(notification);

    // Trigger animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);

    // Remove after 2 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 2000);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Get localized text (fallback to English if translation not available)
 */
function getLocalizedText(key, fallback) {
    // Try to get from existing translation system
    const element = document.getElementById(key);
    if (element && element.textContent) {
        return element.textContent;
    }
    return fallback;
}
