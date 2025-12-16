// index2.js - Task 5 RAG Resume Analyzer (PDF -> embeddings -> retrieve -> dual analysis)
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";
import dotenv from "dotenv";
import * as readline from "readline";
import { createRequire } from "module";

dotenv.config();

const require = createRequire(import.meta.url);
const pdf = require("pdf-parse");

async function analyzeResume(filePath) {
  try {
    console.log(`Reading PDF file from: ${filePath} ...`);

    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdf(dataBuffer);
    const resumeText = pdfData.text;

    if (!resumeText || !resumeText.trim()) {
      throw new Error("No text extracted from PDF (might be image-only PDF).");
    }

    console.log("✅ Extracted resume text from PDF.");

    // LLM + Embeddings
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      temperature: 0.3
    });

    const embeddings = new OpenAIEmbeddings();

    // Split into chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 900,
      chunkOverlap: 150
    });

    const docs = await splitter.createDocuments([resumeText]);

    // Vector store + retriever
    const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
    const retriever = vectorStore.asRetriever();

    // Prompts
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

    // RAG chain builder
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

    console.log("\nGenerating praising analysis...");
    const praising = await praisingChain.invoke({
      question: "Provide a praising analysis of this candidate."
    });

    console.log("Generating critical analysis...");
    const critical = await criticalChain.invoke({
      question: "Provide a critical analysis of this candidate."
    });

    console.log("\n========================================");
    console.log("✅ Hiring Manager Perspective (Pros)");
    console.log("========================================");
    console.log(praising.content);

    console.log("\n========================================");
    console.log("✅ Critical Review (Cons & Gaps)");
    console.log("========================================");
    console.log(critical.content);
  } catch (err) {
    console.error("\n❌ Error during resume analysis:");
    console.error(err);
  }
}

// CLI prompt for file path
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.question("Enter FULL path to a PDF resume: ", (filePath) => {
  const cleaned = filePath.trim().replace(/^"|"$/g, "");
  analyzeResume(cleaned);
  rl.close();
});
