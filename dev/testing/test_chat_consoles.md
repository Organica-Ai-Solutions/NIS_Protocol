# Chat Consoles Manual Test Plan

## Test Execution: October 2, 2025

### Console 1: Classic Chat Console (`/console`)
**URL**: http://localhost:8000/console

#### Test Cases:

1. **Basic Chat Message**
   - Action: Type "Hello" and send
   - Expected: Should receive AI response within 3-5s
   - Status: [ ]

2. **Streaming Response**
   - Action: Type "Count to 5"
   - Expected: Should see streaming tokens appear word by word
   - Status: [ ]

3. **Voice Command**
   - Action: Type "/voice"
   - Expected: Should toggle voice mode with confirmation message
   - Status: [ ]

4. **Research Mode**
   - Action: Enable research mode checkbox, ask "What is AI?"
   - Expected: Should show deeper research results
   - Status: [ ]

5. **Provider Selection**
   - Action: Change provider dropdown, send message
   - Expected: Should use selected provider
   - Status: [ ]

6. **Multimodal (if available)**
   - Action: Upload image or document
   - Expected: Should process and analyze
   - Status: [ ]

7. **Response Formatting**
   - Action: Change output mode/audience level
   - Expected: Response should adapt to settings
   - Status: [ ]

8. **Conversation Memory**
   - Action: Send follow-up question referencing previous message
   - Expected: Should maintain context
   - Status: [ ]

---

### Console 2: Modern Chat (`/modern-chat`)
**URL**: http://localhost:8000/modern-chat

#### Test Cases:

1. **Basic Chat Message**
   - Action: Type "Hello" and send
   - Expected: Should receive AI response within 3-5s
   - Status: [ ]

2. **Streaming Response**
   - Action: Type "Count to 5"
   - Expected: Should see streaming tokens appear word by word
   - Status: [ ]

3. **Voice Command**
   - Action: Type "/voice"
   - Expected: Should toggle conversational voice mode
   - Status: [ ]

4. **Runner Commands**
   - Action: Type "/files"
   - Expected: Should list workspace files
   - Status: [ ]

5. **System Info**
   - Action: Type "/sysinfo"
   - Expected: Should show system information
   - Status: [ ]

6. **Python Execution**
   - Action: Type "/python print('hello')"
   - Expected: Should execute and show output
   - Status: [ ]

7. **File Operations**
   - Action: Type "/read <filename>"
   - Expected: Should read and display file content
   - Status: [ ]

8. **Conversation Memory**
   - Action: Send follow-up question
   - Expected: Should maintain context
   - Status: [ ]

---

## Critical Issues to Check:

### Classic Console:
- [ ] Page loads without errors
- [ ] Input field is functional
- [ ] Send button works
- [ ] Streaming works properly
- [ ] Voice toggle works
- [ ] No JavaScript errors in console
- [ ] Responsive design works on mobile

### Modern Chat:
- [ ] Page loads without errors
- [ ] Input field is functional
- [ ] Send button works
- [ ] Streaming works properly
- [ ] Runner commands execute
- [ ] No JavaScript errors in console
- [ ] Responsive design works on mobile

---

## Performance Checks:
- [ ] Classic console loads in < 2s
- [ ] Modern chat loads in < 2s
- [ ] Chat responses start streaming in < 3s
- [ ] No memory leaks during extended use
- [ ] Smooth animations and transitions

---

## Browser Compatibility:
- [ ] Chrome/Chromium
- [ ] Firefox
- [ ] Safari
- [ ] Edge

---

## Notes:
- Test with different message lengths
- Test with special characters
- Test with code blocks
- Test with markdown formatting
- Test error handling (disconnect backend and try)

