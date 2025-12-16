
import express from "express";
import multer from "multer";
import dotenv from "dotenv";
import fs from "fs";
import { createRequire } from "module";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";

import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { formatDocumentsAsString } from "langchain/util/document";

dotenv.config();

const app = express();
const upload = multer({ dest: "uploads/" });

const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse");

app.use(express.static("public"));

async function runRag(resumeText) {
  const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0.3
  });

  const embeddings = new OpenAIEmbeddings();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 900,
    chunkOverlap: 150
  });

  const docs = await splitter.createDocuments([resumeText]);

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  const retriever = vectorStore.asRetriever();

  const praisingTemplate = `
You are an enthusiastic hiring manager.
Based ONLY on the context below, write a specific summary praising the candidate.
Mention concrete skills, tools, and experience.

Context:
{context}

Question:
{question}

Praising Summary:
`;

  const criticalTemplate = `
You are a cautious, objective recruiter.
Based ONLY on the context below, list weaknesses, gaps, or clarification questions.
If you do not see clear weaknesses, say the resume seems strong overall.

Context:
{context}

Question:
{question}

Critical Analysis:
`;

  const praisingPrompt = PromptTemplate.fromTemplate(praisingTemplate);
  const criticalPrompt = PromptTemplate.fromTemplate(criticalTemplate);

  const createRagChain = (prompt) =>
    RunnableSequence.from([
      {
        context: RunnableSequence.from([
          (input) => input.question,
          retriever,
          formatDocumentsAsString
        ]),
        question: (input) => input.question
      },
      prompt,
      llm
    ]);

  const praisingChain = createRagChain(praisingPrompt);
  const criticalChain = createRagChain(criticalPrompt);

  const praising = await praisingChain.invoke({
    question: "Provide a praising analysis of this candidate."
  });

  const critical = await criticalChain.invoke({
    question: "Provide a critical analysis of this candidate."
  });

  return {
    praising: praising.content,
    critical: critical.content
  };
}

app.post("/analyze", upload.single("resume"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded." });
    }

    const buffer = fs.readFileSync(req.file.path);
    const parsed = await pdfParse(buffer);
    fs.unlinkSync(req.file.path); // delete temp upload

    const resumeText = parsed.text || "";
    if (!resumeText.trim()) {
      return res.status(400).json({
        error: "No text extracted from PDF. (Is it a scanned image PDF?)"
      });
    }

    const result = await runRag(resumeText);
    res.json(result);
  } catch (err) {
    // This will catch quota/billing errors too (which is fine for screenshots)
    res.status(500).json({ error: String(err?.message || err) });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
