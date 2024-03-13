import { Ollama } from "@langchain/community/llms/ollama";
import { PromptTemplate } from "@langchain/core/prompts";

const ollama = new Ollama({
  baseUrl: "http://localhost:11434", // Default value
  model: "llama2", // Default value
});


const promptTemplate = PromptTemplate.fromTemplate(
    "Tell me a joke about {topic}"
);

console.log(await promptTemplate.format({ topic: "bears" }))

const stream = await ollama.stream(await promptTemplate.format({ topic: "bears" }));

const chunks = [];
for await (const chunk of stream) {
  chunks.push(chunk);
}

console.log(chunks.join(""));