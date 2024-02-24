import time

import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

llm = Ollama(model='llama2', base_url='http://localhost:11434',
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
             , temperature=0.5)


# Streamed response emulator
def response_generator(question):
    template = """
    
    <<SYS>> Você é um assistente pessoal de AI. <</SYS>>
    [INST] Responda todas as perguntas de forma simples e objetivo no idoma português
        {question} 
    [/INST]
    
    """

    prompt_template = PromptTemplate(input_variables=['question'], output_parser=None, partial_variables={},
                                     template=template,
                                     template_format='f-string', validate_template=True)

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response_llm = llm_chain.run({"question": question})

    for word in response_llm.split():
        yield word + " "
        time.sleep(0.05)


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
