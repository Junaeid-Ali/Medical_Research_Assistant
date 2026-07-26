from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore=Chroma(
    persist_directory="Medical-db",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
        "k":5,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatMistralAI(model="mistral-small-2506")



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are MedRAG, an AI-powered Medical Research Assistant.

Your role is to answer users' questions using ONLY the retrieved medical research documents provided to you.

Rules:
1. Use ONLY the retrieved context to answer.
2. Never use your own knowledge.
3. Never fabricate or guess information.
4. If the answer is not present in the retrieved documents, say:
   "I could not find sufficient evidence in the retrieved medical documents to answer this question."
5. If multiple papers contain different findings, mention all viewpoints objectively.
6. Summarize information from multiple papers when appropriate.
7. Keep the answer scientifically accurate, concise, and easy to understand.
8. Never provide personal medical advice or make clinical recommendations.
9. Clearly mention uncertainty if the evidence is limited.
10. At the end of every response, include a Sources section.

Response Format:

## Summary
A concise answer.

## Detailed Explanation
Explain the findings using the retrieved evidence.

## Sources
List the paper titles (and page numbers if available).
"""
        ),

        (
            "human",
            """
Retrieved Context:

{context}

----------------------------------------

User Question:

{question}
"""
        )
    ]
)

print("Medical RAG System Created")
print("Press 0 to exit")

while True:
    query=input("YOU:")
    if query=="0":
        break

    docs=retriever.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })

    response=llm.invoke(final_prompt)

    print(f"AI:{response.content}")
