const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const os = require('os');
const cors = require('cors');
const OpenAI = require('openai');

// Load environment variables (used for local testing, ignored on Render)
require('dotenv').config();

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
// Serve the index.html frontend to the browser
app.use(express.static(path.join(__dirname))); 

// Setup Multer to handle audio file uploads from the frontend
const upload = multer({ dest: os.tmpdir() });

// Initialize OpenAI (Make sure you set OPENAI_API_KEY in Render.com's environment variables)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, 
  maxRetries: 3,
});

// Memory
let conversationHistory = [];

const SYSTEM_PROMPT = `YYou are my REAL-TIME INTERVIEW COPILOT.

I am in a LIVE interview. Give answers exactly how a confident engineer speaks — short, clear, and practical.

---------------------------------------
🎯 CORE RULE (MOST IMPORTANT):

Every answer MUST follow this structure:

1. DIRECT ANSWER (1–2 lines only)
→ Straight to the point. No theory dumping.

2. MY EXPERIENCE (2–4 lines)
→ Connect to my real project (Kotak Life).
→ Show that I have actually built it.

3. SIMPLE EXPLANATION (only if needed)
→ Explain in very simple words.
→ No textbook language.

---------------------------------------
🧠 HOW I SPEAK (VERY IMPORTANT):

- Use simple English
- Sound natural, not robotic
- Use phrases like:
  "So basically..."
  "In my case..."
  "What I did was..."
  "One challenge I saw was..."

- NEVER sound like ChatGPT
- NEVER give long paragraphs
- NEVER over-explain

---------------------------------------
🚫 STRICTLY AVOID:

- Definitions like textbook
- Long paragraphs
- Buzzwords like:
  leveraged, utilized, orchestrated, seamless

- No bullet points
- No formatting
- No markdown

---------------------------------------
💼 MY PROFILE (USE IN EVERY ANSWER):

I am Shashi, Senior AI Engineer at Kotak Life.

I build:
- AI underwriting automation (Textract + LLM + Step Functions)
- RAG systems (LangChain + Pinecone)
- Event-driven architectures on AWS

---------------------------------------
📌 PROJECT MAPPING (MANDATORY):

Whenever possible, connect answer to:

1. Underwriting AI System
→ OCR + LLM + rule engine
→ Step Functions + Lambda + SageMaker

2. Sales AI Assistant (RAG)
→ Embeddings + Pinecone + real-time recommendations

---------------------------------------
🎯 GOAL:

Interviewer should feel:
"This guy has actually built systems"

---------------------------------------
EXAMPLE STYLE:

Q: What is RAG?

Answer:
RAG is basically a way to give LLM external knowledge instead of relying only on its training.

In my case, I used it in a Sales AI assistant where we stored Kotak brochures in Pinecone and retrieved relevant data before sending to the model.

So instead of guessing, the model answers based on actual company data.

---------------------------------------

Now wait for my question.`;

// Endpoint 1: Clear Memory (When you hit the refresh button)
app.post('/api/reset', (req, res) => {
    conversationHistory = [];
    res.json({ success: true });
});

// Endpoint 2: Process Audio (When you finish recording)
app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No audio file provided" });
        }
        
        // Whisper requires a file extension to recognize the format
        const tempFilePath = req.file.path + '.webm';
        fs.renameSync(req.file.path, tempFilePath);

        const transcription = await openai.audio.transcriptions.create({
            file: fs.createReadStream(tempFilePath),
            model: 'whisper-1',
            language: 'en'
        });

        // Delete the temporary audio file
        fs.unlinkSync(tempFilePath);

        res.json({ text: transcription.text });
    } catch (error) {
        console.error("Transcription error:", error);
        res.status(500).json({ error: "Transcription failed" });
    }
});

// Endpoint 3: Stream the AI Answer
app.post('/api/ask', async (req, res) => {
    const { question } = req.body;
    
    if (!question) {
        return res.status(400).json({ error: "Question is required" });
    }

    // Prepare headers for Server-Sent Events (Real-time Streaming)
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Add user question to memory
    conversationHistory.push({ role: "user", content: question });
    
    // Safeguard: Keep only the last 6 messages
    if (conversationHistory.length > 6) {
        conversationHistory = conversationHistory.slice(conversationHistory.length - 6);
    }

    try {
        const stream = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: SYSTEM_PROMPT },
                ...conversationHistory
            ],
            temperature: 0,
            max_tokens: 500,
            stream: true
        });

        let fullAnswer = "";
        
        // Push words to the frontend instantly as they generate
        for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content || "";
            if (content) {
                fullAnswer += content;
                res.write(`data: ${JSON.stringify({ text: content })}\n\n`);
            }
        }
        
        // Save final answer to memory
        if (fullAnswer.trim().length > 0) {
            conversationHistory.push({ role: "assistant", content: fullAnswer });
        }
        
        // Signal the frontend that the stream is finished
        res.write(`data: [DONE]\n\n`);
        res.end();

    } catch (error) {
        console.error("OpenAI Error:", error);
        res.write(`data: ${JSON.stringify({ error: "Failed to generate answer" })}\n\n`);
        res.end();
    }
});

// Start the server (Render will assign the PORT automatically)
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Web server running on port ${PORT}`);
});
