import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

class ChatBot():
    load_dotenv()

    def __init__(self, flavor):
        if flavor == "openai":
            embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(
               model="gpt-3.5-turbo",
                temperature=0.5
            )
        elif flavor == "huggingface":
            embeddings = HuggingFaceEmbeddings()
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=1,
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )
        else:
            embeddings = None
            llm = None

        vector_store = FAISS.load_local("./data/nebraska-code", embeddings, flavor, allow_dangerous_deserialization=True)

        template = """
            You are a helpful assistant helping attendees at Nebraska.Code(), a software development conference, make the most of their conference by answering questions about sessions and helping them find the best sessions to attend based on their interests. Your answers should be conversational - do not simply respond with the verbatim session data, but do make sure to include relevant information such as the room and times of the sessions. If there is more than one session that matches their query, feel free to reference them all. A bulleted list is acceptable in the output.
            Here are some sessions that appear relevant to the user's question. Use these as you see fit to help answer the user's question:

            {context}

            Question: {question}
            Answer: 

        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.chain = (
            {
                "context": vector_store.as_retriever(search_kwargs={'k': 10}),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def get_chain(self):
        return self.chain
