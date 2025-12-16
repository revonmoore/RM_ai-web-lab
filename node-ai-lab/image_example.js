// Task 3 - Image generation example with OpenAI
import OpenAI from "openai";
import dotenv from "dotenv";
import fs from "fs";

dotenv.config();

// Uses the same API key from .env
const client = new OpenAI();

async function main() {
  try {
    console.log("Requesting image from OpenAI...");

    const prompt =
      "A futuristic Kean University campus at sunset, minimalist digital art style";

    const result = await client.images.generate({
      model: "gpt-image-1",
      prompt,
      size: "1024x1024"
    });

    // Get base64 image and save as PNG
    const imageBase64 = result.data[0].b64_json;
    const buffer = Buffer.from(imageBase64, "base64");
    const filename = "kean_future.png";

    fs.writeFileSync(filename, buffer);

    console.log(`Image saved as ${filename}`);
  } catch (error) {
    console.error("Error generating image:", error);
  }
}

main();
