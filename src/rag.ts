// From https://js.langchain.com/docs/get_started/quickstart

//Import Lib
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

const outputParser = new StringOutputParser();

// const prompt = ChatPromptTemplate.fromMessages([
//     ["system", "You are a world class technical documentation writer."],
//     ["user", "{input}"],
// ]);

const chatModel = new ChatOllama({
    baseUrl: "http://localhost:11434", // Default value
    model: "llama2",
});

// const response = await chatModel.invoke("What is Palo IT");
// console.log(response)
// const chain = prompt.pipe(chatModel).pipe(outputParser);

// const response = await chain.invoke({
//     input: "What is Palo IT",
// });

// console.log(response)
// Chat model


//1. Create folder 
// 2. Find a nice pdf 
// Load Document
const loader = new TextLoader("./data/sample.txt");
const docs = await loader.load();
console.log({ docLen: docs.length });
// 3. Split Documents
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 100,
});

const splitDocs = await textSplitter.splitDocuments(docs);

const avgDocLength = (documents: Document<Record<string, any>>[]): number => {
    return documents.reduce((sum, doc) => sum + doc.pageContent.length, 0) / documents.length;
};
const avgCharCountPre = avgDocLength(docs);
const avgCharCountPost = avgDocLength(splitDocs);

console.log(`Average length among ${docs.length} documents loaded is ${avgCharCountPre} characters.`);
console.log(`After the split we have ${splitDocs.length} documents more than the original ${docs.length}.`);
console.log(`Average length among ${docs.length} documents (after split) is ${avgCharCountPost} characters.`);

//Set up Vector Store
// 1. Create Embedding Object
const embeddings = new OllamaEmbeddings({
    model: "llama2",
    maxConcurrency: 5,
});
// 2. Create Vector Store Object


const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);


//Stuff Doc Chain
const prompt =
    ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
});

const retriever = vectorstore.asRetriever();

// Retrieve Chain
const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
});

const result = await retrievalChain.invoke({
    input: "what is Tech Edge?",
});

console.log("================================\n\n\n");
console.log(result.answer);