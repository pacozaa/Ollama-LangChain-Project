// From https://js.langchain.com/docs/get_started/quickstart
// https://huggingface.co/learn/cookbook/advanced_rag
//Import Lib
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import {
    AzureAISearchVectorStore,
    AzureAISearchQueryType,
} from "@langchain/community/vectorstores/azure_aisearch";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

const chatModel = new ChatOllama({
    baseUrl: "http://localhost:11434", // Default value
    model: "llama2",
});

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
// const embeddings = new OllamaEmbeddings({
//     model: "llama2",
//     maxConcurrency: 5,
// });
// Alternative Embedding, Faster?
const embeddings = new HuggingFaceTransformersEmbeddings({
    modelName: "Xenova/all-MiniLM-L6-v2",
    maxConcurrency: 3
});

// 2. Create Vector Store Object

//https://learn.microsoft.com/en-us/azure/search/search-query-lucene-examples#example-1-fielded-search
const vectorstore = await AzureAISearchVectorStore.fromDocuments(
    splitDocs,
    embeddings,
    {
        indexName: 'coolVector',
        search: {
            type: AzureAISearchQueryType.SimilarityHybrid,
        },
    }
);


console.log({vectorstore})

//Stuff Doc Chain
const prompt =
    ChatPromptTemplate.fromTemplate(`
    Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);
//Basic Chain
const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
});

const retriever = vectorstore.asRetriever({verbose: true});//4 fetch top 4 similarity result 

// Retrieve Chain
const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
});

const result = await retrievalChain.invoke({
    input: "what is Tech Edge?",
});

console.log("================================\n\n\n");
console.log(result);