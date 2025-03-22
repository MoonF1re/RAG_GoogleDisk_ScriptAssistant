from langchain_community.chat_models import ChatOllama #Библиотека да Ламы
from hybrid_context_extractor import hybrid_extract_context
from langchain_core.prompts import ChatPromptTemplate #Позволяет создавать шаблон для промта
from langchain_core.prompts import MessagesPlaceholder #Добавлет место куда можно подставить историю чата
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from chroma_utils import vectorstore

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
#vectorstore преобразуется в retriever – компонент, который по запросу будет возвращать релевантные фрагменты документов.


#Инструкция для LLM
contextualize_q_system_prompt = (
     "Given a chat history and the latest user question "
     "which might reference context in the chat history, "
     "formulate a standalone question which can be understood "
     "without the chat history. Do NOT answer the question, "
     "just reformulate it if needed and otherwise return it as is."
 )
""" Этот текст инструктирует модель взять вопрос, который может быть неясен без контекста, 
    и переписать его так, чтобы он стал самостоятельным.
"""


#Формат промта для LLM чтобы он сделал вопрос самостоятельным
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

#Формат Финального промта для LLM
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI script assistant. Use the following context to answer the user's question."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

class HybridRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def __call__(self, query, chat_history=""):
        # Если query пришёл как словарь, извлекаем текст из поля "input"
        if isinstance(query, dict):
            query = query.get("input", "")
        # Получаем первоначальные результаты от базового retriever
        initial_chunks = self.base_retriever.invoke({"input": query, "chat_history": chat_history})
        # Преобразуем каждый элемент в строку: если это объект Document (или dict) с полем page_content, берем его
        texts = []
        for chunk in initial_chunks:
            if isinstance(chunk, dict):
                texts.append(chunk.get("page_content", str(chunk)))
            elif hasattr(chunk, "page_content"):
                texts.append(chunk.page_content)
            else:
                texts.append(str(chunk))

        # Применяем гибридное извлечение для уточнения контекста
        refined_chunks = hybrid_extract_context(query, texts)

        # Сохраняем объединённый контекст
        self.last_context = "\n\n".join(refined_chunks)

        # Оборачиваем каждую строку в объект Document с пустыми метаданными
        refined_docs = [Document(page_content=txt, metadata={}) for txt in refined_chunks]
        return refined_docs

    def similarity_search(self, query: str, **kwargs):
        # Если есть chat_history в kwargs, извлекаем его, иначе – пустая строка
        chat_history = kwargs.get("chat_history", "")
        return self.__call__(query, chat_history)

    def with_config(self, **kwargs):
        # Если нужна дополнительная конфигурация, здесь можно её добавить
        return self




def get_rag_chain(model="llama3.2"):
    llm = ChatOllama(model=model)

    #Базовый ретривер
    base_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    #Гибридный
    hybrid_retriever_obj = HybridRetriever(base_retriever)

    # Создаем цепочку для генерации ответа на основе финального промта
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Собираем всю цепочку
    rag_chain = create_retrieval_chain(hybrid_retriever_obj, question_answer_chain)

    #СТАРАЯ ЦЕПОЧКА
    #rag_chain = create_retrieval_chain(base_retriever, question_answer_chain)
    return rag_chain, hybrid_retriever_obj
