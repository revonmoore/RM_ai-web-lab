import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const client = new OpenAI();

const PROMPT =
  "Explain the difference between machine learning and traditional programming.";

const MODELS = [
  "gpt-4o-mini",
  "gpt-4o"
];

async function compareModels() {
  console.log("=== NON-STREAMING COMPARISON ===\n");

  for (const model of MODELS) {
    console.log(`--- Model: ${model} ---\n`);

    const completion = await client.chat.completions.create({
      model,
      messages: [{ role: "user", content: PROMPT }]
    });

    console.log(completion.choices[0].message.content.trim());
    console.log("\n--------------------------------------------\n");
  }
}

async function streamingExample() {
  console.log("\n=== STREAMING EXAMPLE (gpt-4o-mini) ===\n");

  const stream = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: PROMPT }],
    stream: true
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }

  console.log("\n\n=== END OF STREAM ===");
}

async function main() {
  await compareModels();
  await streamingExample();
}

main().catch(console.error);
