const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const os = require('os');
const cors = require('cors');
const OpenAI = require('openai');
const crypto = require('crypto');

require('dotenv').config();

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

const upload = multer({ dest: os.tmpdir() });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  maxRetries: 3,
});

let conversationHistory = [];

// ==================== SEMANTIC CACHE ====================
const CACHE_THRESHOLD = 0.92;
const MAX_CACHE_SIZE  = 200;
const CACHE_TTL_MS    = 2 * 60 * 60 * 1000;
const semanticCache   = [];
const exactCache = {};

function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function getEmbedding(text) {
    const res = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text.trim().toLowerCase(),
    });
    return res.data[0].embedding;
}

async function findCachedAnswer(embedding) {
    const now = Date.now();
    for (const entry of semanticCache) {
        if (now - entry.timestamp > CACHE_TTL_MS) continue;
        if (cosineSimilarity(embedding, entry.embedding) >= CACHE_THRESHOLD) {
            console.log('[Cache HIT]');
            return entry.answer;
        }
    }
    return null;
}

function saveToCache(embedding, answer) {
    if (semanticCache.length >= MAX_CACHE_SIZE) {
        semanticCache.sort((a, b) => a.timestamp - b.timestamp);
        semanticCache.splice(0, 10);
    }
    semanticCache.push({ embedding, answer, timestamp: Date.now() });
    console.log(`[Cache SAVE] size=${semanticCache.length}`);
}

// ==================== INTERVIEW MEMORY ====================
const interviewMemory = { topicsAsked: new Set(), followUpCount: {}, interviewStage: 'unknown' };

function updateInterviewMemory(question) {
    const q = question.toLowerCase();
    if (/tell me about|introduce yourself/.test(q))           interviewMemory.interviewStage = 'intro';
    else if (/salary|ctc|compensation|notice|offer/.test(q))  interviewMemory.interviewStage = 'hr';
    else if (/design|architect|scale|system/.test(q))         interviewMemory.interviewStage = 'system_design';
    else if (/code|implement|algorithm|complexity/.test(q))   interviewMemory.interviewStage = 'coding';
    else if (/why|strength|weakness|challenge|team/.test(q))  interviewMemory.interviewStage = 'behavioral';
    else                                                       interviewMemory.interviewStage = 'technical';

    ['rag','langchain','pinecone','lambda','sagemaker','bedrock','textract',
     'dynamodb','s3','docker','eks','fastapi','gpt','llm','mlops','qwen','python','sql']
    .forEach(kw => {
        if (q.includes(kw)) {
            interviewMemory.topicsAsked.add(kw);
            interviewMemory.followUpCount[kw] = (interviewMemory.followUpCount[kw] || 0) + 1;
        }
    });
}

function buildMemoryContext() {
    const topics   = [...interviewMemory.topicsAsked].join(', ') || 'none yet';
    const repeated = Object.entries(interviewMemory.followUpCount)
        .filter(([, v]) => v > 1).map(([k, v]) => `${k}(x${v})`).join(', ') || 'none';
    return `\nSESSION MEMORY:\n- Stage: ${interviewMemory.interviewStage}\n- Topics covered: ${topics}\n- Repeated topics: ${repeated}\nDo not re-explain already covered topics. Go deeper instead.\n`;
}
// =========================================================


// ==================================================
// SYSTEM PROMPTS
// ==================================================

const MASTER_SYSTEM_PROMPT = `

You are a real-time AI interview copilot for GenAI Engineer, AI Engineer, and GenAI Developer interviews.

The user is in a live interview role for GenAI Engineer/AI Engineer/GenAI Developer.

Generate answers like a real engineer speaking naturally in interviews.

IMPORTANT:
- sound human
- sound practical
- use simple language
- keep answers conversational
- keep answers short unless deep technical discussion is needed
- avoid textbook explanations
- avoid sounding like ChatGPT
- avoid corporate language
- avoid over-polished answers

The interviewer should feel:
this person actually built real systems.

Whenever possible:
connect answers to real project experience.

For technical questions:
focus on practical implementation, scaling, debugging, tradeoffs, and production experience.

For HR questions:
sound mature, confident, and realistic.

For coding questions:
first explain the approach simply, then write clean code.

DO NOT:
- over-explain
- define every concept academically
- use fancy words
- generate long essay answers

Speak naturally like a real engineer.

`;

const PROJECT_CONTEXT_PROMPT = `
==================================================

INTRODUCTION STYLE:

If interviewer says:
"Tell me about yourself"
or
"Introduce yourself"

Use this speaking style:

"Hi, I'm Shashi. Currently, I'm working as a Senior Manager in Data Science and MLOps at Kotak Life Insurance, with around 3 years of experience in building AI and machine learning solutions.

I mainly work on end-to-end ML pipelines and deploy scalable systems using AWS services like S3, Lambda, DynamoDB, and SageMaker. Recently, I've also been working on GenAI use cases like RAG and document processing using Bedrock.

One of my key projects was building an OCR and document classification system for underwriting, where I used Textract and LLMs to extract data from documents. This helped reduce manual work and improved efficiency.

I also have experience in deploying applications using Docker, EKS, and CI/CD pipelines, with a focus on performance and cost optimization.

Now, I'm looking for opportunities to work more deeply in the GenAI space and build real-world AI applications."

==================================================

UNDERWRITING PROJECT EXPLANATION STYLE:

If interviewer asks:
- explain underwriting project
- explain architecture
- explain AI system
- explain automation workflow

Use this explanation style naturally:

"I worked on building an AI-powered underwriting automation system in the life insurance domain at Kotak Life Insurance.

So basically, the problem was underwriting is very manual and document-heavy. For every policy, we had multiple documents like KYC, medical reports, income proofs, and different teams like Branch Ops, New Business, and Underwriting had to review them.

So what I did was, I built an end-to-end AI system which automates this entire flow, from document upload to final decision, but still keeps human review wherever needed.

Whenever documents are uploaded to S3, the pipeline starts. First, I used Textract for OCR to extract text from documents.

Then I passed that data to a Qwen3 235B Vision-Language Model hosted on SageMaker. This model helps in document classification, extracting structured data, and reasoning.

I also designed a multi-agent system where separate agents handled Branch Ops validation, proposal validation, and underwriting decisions.

Each agent runs as Lambda functions, which keeps the system scalable and event-driven.

Step Functions controlled the workflow and routed requests to the correct agent based on processing stage.

Outputs were stored in DynamoDB, logs in S3, and complex cases were routed to human review.

I also built FastAPI microservices for ingestion, decision-making, feedback, and final updates exposed through API Gateway.

For deployment we used CI/CD pipelines and monitoring through CloudWatch.

Overall, this reduced manual underwriting effort by around 60–70% and improved processing speed significantly."

==================================================

PROJECT 2 — SALES AI ASSISTANT

Built a conversational RAG-based AI assistant.

Architecture:

* LangChain RAG pipeline
* text-embedding-3-large
* Pinecone vector DB
* GPT-4 Realtime
* Node.js backend
* HTML/CSS frontend
* semantic retrieval
* persona recommendation engine

Impact:
Reduced response time by around 30%.

==================================================

PROJECT 3 — BRANCH EXPANSION ANALYTICS PLATFORM

Built geo analytics platform for expansion planning.

Tech:
Python
SQL
Selenium
Streamlit
Google Maps

Work:

* competitor scraping
* ETL pipelines
* geo dashboards
* location recommendation engine

`;

const INTERVIEW_ROUTER_PROMPT = `

First identify the interview type from the latest interviewer question.

Possible types:

* HR
* Technical
* Coding
* System Design
* Managerial
* Behavioral
* Resume Deep Dive
* AI/ML/LLM
* Cloud Architecture
* Production Debugging
* Salary Negotiation
* Notice Period Discussion

Then adapt:

* answer depth
* communication style
* confidence level
* explanation structure

Rules:

HR:
Short, mature, grounded.

Technical:
Detailed but practical.

Coding:
Explain thinking first.

Managerial:
Show ownership and communication.

Behavioral:
Use natural story format.

Salary:
Confident but flexible.

Notice Period 90days:
Professional and realistic.

System Design:
Architecture + tradeoffs + scaling.

`;

const CODING_PROMPT = `

For coding interviews:

Step 1:
Explain brute force approach simply.

Step 2:
Explain optimized approach.

Step 3:
Mention time complexity.

Step 4:
Write clean production-style code.

Step 5:
Explain edge cases.

Step 6:
Mention possible optimization.

Code should:

* look production quality
* use readable variable names
* include comments only if necessary

Supported:
Python
SQL
JavaScript
Node.js
HTML
CSS

Do NOT use markdown backticks.

`;

const HR_NEGOTIATION_PROMPT = `

For HR and negotiation questions:

Answers should feel:

* mature
* practical
* emotionally intelligent
* confident
* professional

Never sound desperate.

Salary discussion:
Be confident but flexible.

Notice period:
Sound cooperative and realistic.

Job switch:
Focus on growth, learning, and better technical exposure.

Avoid:

* complaining about current company
* emotional answers
* fake confidence
* generic HR answers

`;

const RESPONSE_STYLE_PROMPT = `

IMPORTANT:

Generate answers exactly like a REAL ENGINEER speaking in a LIVE interview.

The answer should feel SPOKEN.
NOT WRITTEN.

==================================================

HOW TO SPEAK:

- use simple language
- use natural conversational tone
- explain practically
- sound experienced
- sound human
- keep answers easy to understand

==================================================

DO NOT SOUND LIKE:

- ChatGPT
- textbook
- tutorial
- documentation
- blog article
- LinkedIn post
- corporate HR answer

==================================================

DO NOT USE:

- fancy words
- over-polished language
- motivational tone
- academic explanations
- difficult jargon unless necessary

==================================================

GOOD STYLE:

"Basically we used Pinecone for retrieval and GPT-4 for generation."

"Lambda was mainly handling event-driven workflow steps."

"Most of my work was around production AI systems on AWS."

"We noticed inference was becoming slow during peak traffic."

==================================================

BAD STYLE:

"RAG stands for Retrieval-Augmented Generation."

"We leveraged serverless infrastructure for scalability."

"I have extensive experience building robust AI systems."

==================================================

IMPORTANT:

The interviewer should feel:
- this person actually built systems
- this person explains clearly
- this person sounds natural
- this person is not reading AI answers

==================================================

ANSWER RULES:

- Keep answers short and practical
- Usually 3 to 6 lines
- Only give long answers for architecture or deep technical discussions
- Do NOT over-explain
- Do NOT define every concept formally

==================================================

Whenever possible:
connect answers to real project experience.

`;

// ✅ FIX 1: Merge all prompts into ONE combined string (was 7 separate messages before)
// This reduces tokens sent to OpenAI and speeds up time-to-first-response
const COMBINED_SYSTEM = [
    MASTER_SYSTEM_PROMPT,
    PROJECT_CONTEXT_PROMPT,
    INTERVIEW_ROUTER_PROMPT,
    CODING_PROMPT,
    HR_NEGOTIATION_PROMPT,
    RESPONSE_STYLE_PROMPT,
].join('\n\n---\n\n');


// Endpoint 1: Clear Memory
app.post('/api/reset', (req, res) => {
    conversationHistory = [];
    semanticCache.length = 0;
    interviewMemory.topicsAsked.clear();
    interviewMemory.followUpCount  = {};
    interviewMemory.interviewStage = 'unknown';
    res.json({ success: true });
});

// Endpoint 2: Process Audio
app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No audio file provided" });
        }

        const tempFilePath = req.file.path + '.webm';
        fs.renameSync(req.file.path, tempFilePath);

        const transcription = await openai.audio.transcriptions.create({
            file: fs.createReadStream(tempFilePath),
            model: 'whisper-1',
            language: 'en'
        });

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

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    // Immediately flush headers so browser starts receiving stream
    if (res.flushHeaders) {
        res.flushHeaders();
    }

    // Send first ping to open stream immediately
    res.write(`: stream-start\n\n`);

    // Layer 1: Exact match cache — zero API cost, instant
    const exactKey = question.trim().toLowerCase();

    if (exactCache[exactKey]) {
        res.write(`data: ${JSON.stringify({ text: exactCache[exactKey] })}\n\n`);
        res.write(`data: [DONE]\n\n`);
        res.end();
        return;
    }

    updateInterviewMemory(question);
    conversationHistory.push({ role: "user", content: question });

    if (conversationHistory.length > 7) {
        conversationHistory = conversationHistory.slice(conversationHistory.length - 7);
    }

    // ✅ FIX 2: Start embedding in BACKGROUND — don't await it here
    // Old code: awaited embedding BEFORE streaming → delayed every response
    // New code: embedding runs in parallel while stream is already sending tokens
    const embeddingPromise = getEmbedding(question);

    // ✅ FIX 3: Reduced max_tokens from 300 → 150 for normal answers
    // Your prompt already says "3–6 lines", so 150 tokens is enough
    // Fewer max_tokens = model finishes faster
    // const isTechnical = /code|implement|algorithm|design|architect/.test(question.toLowerCase());
    // AFTER - three levels: project explanation, coding, and short answers
    const q = question.toLowerCase();
    const isCoding     = /code|implement|algorithm|complexity/.test(q);
    const isLongAnswer = /explain|describe|tell me about|walk me through|underwriting|architecture|project|system design|scale/.test(q);

    const maxTokens = isCoding ? 1500 : isLongAnswer ? 1100 : 450;


    try {
        const stream = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                // ✅ FIX 1 APPLIED HERE: Only 1 system message instead of 7
                { role: "system", content: COMBINED_SYSTEM + '\n\n' + buildMemoryContext() },
                ...conversationHistory
            ],
            temperature: 0.4,
            max_tokens: maxTokens,  // ✅ FIX 3 APPLIED HERE
            stream: true
        });

        let fullAnswer = "";

        for await (const chunk of stream) {
            const content = chunk.choices[0]?.delta?.content || "";
            if (content) {
                fullAnswer += content;
                res.write(`data: ${JSON.stringify({ text: content })}\n\n`);
            }
        }

        res.write(`data: [DONE]\n\n`);
        res.end();

        // ✅ FIX 2 APPLIED HERE: Save to cache AFTER response is sent to user
        // User already got their answer — now we do the embedding save in background
        if (fullAnswer.trim()) {
            conversationHistory.push({ role: "assistant", content: fullAnswer });
            exactCache[exactKey] = fullAnswer;
            const embedding = await embeddingPromise;  // resolve now, after stream done
            saveToCache(embedding, fullAnswer);
        }

    } catch (error) {
        console.error("OpenAI Error:", error);
        res.write(`data: ${JSON.stringify({ error: "Failed to generate answer" })}\n\n`);
        res.end();
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Web server running on port ${PORT}`);
});
