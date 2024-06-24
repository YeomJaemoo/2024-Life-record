import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.schema.messages import HumanMessage, AIMessage
import tiktoken
import json
import base64

def main():
    st.set_page_config(page_title="kangsinchat", page_icon="🏫")
    st.image('knowhow.png')
    st.title("_강신중학교 :red[생활기록부기재요령 도우미]_ 🏫")
    st.header("😶주의!이 챗봇은 참고용으로 사용하세요!", divider='rainbow')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        folder_path = Path()
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        model_name = 'gpt-3.5-turbo'
        
        st.text("아래의 'Process'를 누르고\n아래 채팅창이 활성화 될 때까지\n잠시 기다려주세요!🙂🙂🙂")
        process = st.button("Process")
        
        if process:
            files_text = get_text_from_folder(folder_path)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key, model_name)
            st.session_state.processComplete = True

        if st.session_state.conversation and st.session_state.chat_history:
            save_button = st.button("대화 저장")
            if save_button:
                save_conversation(st.session_state.chat_history)
    
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                         "content": "생활기록부기재요령에 대해 물어보세요!😊"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("생각 중..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.get('chat_history'):
        st.write("## 채팅 내용")
        for idx, message in enumerate(st.session_state.chat_history):
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            content = message.content
            st.write(f"### 메시지 {idx + 1}")
            st.write(f"**역할:** {role}")
            st.write(f"**내용:** {content}")
            st.write("---")
            
            # 메시지 다운로드 링크 생성
            msg = f"역할: {role}\n내용: {content}\n"
            b64 = base64.b64encode(msg.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="채팅_메시지_{idx + 1}.txt">메시지 다운로드</a>'
            st.markdown(href, unsafe_allow_html=True)

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_from_folder(folder_path):
    doc_list = []
    folder = Path(folder_path)
    files = folder.iterdir()

    for file_path in files:
        if file_path.is_file():
            if file_path.suffix == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load_and_split()
            elif file_path.suffix == '.docx':
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load_and_split()
            elif file_path.suffix == '.pptx':
                loader = UnstructuredPowerPointLoader(str(file_path))
                documents = loader.load_and_split()
            else:
                documents = []
            doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type='mmr'),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

def save_conversation(chat_history):
    conversation = []
    for message in chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
        conversation.append({"role": role, "content": content})
    
    conversation_json = json.dumps(conversation, ensure_ascii=False, indent=4)
    st.download_button(
        label="대화 다운로드",
        data=conversation_json,
        file_name="conversation.json",
        mime="application/json"
    )

if __name__ == '__main__':
    main()
