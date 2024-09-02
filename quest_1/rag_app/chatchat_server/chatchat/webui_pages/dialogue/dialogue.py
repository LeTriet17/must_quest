import base64
import uuid
import os
import re
import time
from typing import List, Dict

import streamlit as st
from streamlit_antd_components.utils import ParseItems

import openai
from streamlit_chatbox import *
from streamlit_modal import Modal
from datetime import datetime

from chatchat.configs import (LLM_MODEL_CONFIG, SUPPORT_AGENT_MODELS, MODEL_PLATFORMS)
from chatchat.server.callback_handler.agent_callback_handler import AgentStatus
from chatchat.server.utils import MsgType, get_config_models
from chatchat.server.utils import get_tool_config
from chatchat.webui_pages.utils import *
img_dir = (Path(__file__).absolute().parent.parent.parent)

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        img_dir,
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)

def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    Get message history.
    content_in_expander controls whether to return the content inside the expander element, 
    generally, it can be selected when exporting, not needed for LLM's history.
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)

# @st.cache_data
# def upload_temp_docs(files, _api: ApiRequest) -> str:
#     '''
#     Upload files to a temporary directory for file dialogue.
#     Returns the temporary vector library ID.
#     '''
#     return _api.upload_temp_docs(files).get("data", {}).get("id")

def parse_command(text: str, modal: Modal) -> bool:
    '''
    Check if the user has entered a custom command. Currently supported commands are:
    /new {session_name}. If the name is not provided, defaults to "Session X".
    /del {session_name}. If the name is not provided and there are >1 sessions, deletes the current session.
    /clear {session_name}. If the name is not provided, clears the current session by default.
    /stop {session_name}. If the name is not provided, stops the current session by default.
    /list_sessions. Lists all available sessions.
    /switch {session_name}. Switches to the specified session.
    /help. View command help.
    Returns: True if the input is a command, otherwise False.
    '''
    if m := re.match(r"/([^\s]+)\s*(.*)", text):
        cmd, name = m.groups()
        name = name.strip()
        conv_names = chat_box.get_chat_names()
        if cmd == "help":
            cmds = [x for x in parse_command.__doc__.split("\n") if x.strip().startswith("/")]
            chat_box.ai_say("\n\n".join(cmds))
        elif cmd == "new":
            if not name:
                i = 1
                while True:
                    name = f"Session {i}"
                    if name not in conv_names:
                        break
                    i += 1
            if name in st.session_state["conversation_ids"]:
                st.error(f"This session name “{name}” already exists")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"][name] = uuid.uuid4().hex
                st.session_state["cur_conv_name"] = name
        elif cmd == "del":
            name = name or st.session_state.get("cur_conv_name")
            if len(conv_names) == 1:
                st.error("This is the last session and cannot be deleted")
                time.sleep(1)
            elif not name or name not in st.session_state["conversation_ids"]:
                st.error(f"Invalid session name: “{name}”")
                time.sleep(1)
            else:
                st.session_state["conversation_ids"].pop(name, None)
                chat_box.del_chat_name(name)
                st.session_state["cur_conv_name"] = ""
        elif cmd == "clear":
            chat_box.reset_history(name=name or None)
        elif cmd == "list_sessions":
            chat_box.ai_say("Available sessions: " + ", ".join(conv_names))
        elif cmd == "switch":
            if name in conv_names:
                st.session_state["cur_conv_name"] = name
            else:
                st.error(f"Invalid session name: “{name}”")
                time.sleep(1)
        return True
    return False


def dialogue_page(api: ApiRequest, is_lite: bool = False):

    st.session_state.setdefault("conversation_ids", {})
    st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
    st.session_state.setdefault("file_chat_id", None)

    # Pop-up custom command help information
    modal = Modal("Custom Commands", key="cmd_help", max_width="500")
    if modal.is_open():
        with modal.container():
            cmds = [x for x in parse_command.__doc__.split("\n") if x.strip().startswith("/")]
            st.write("\n\n".join(cmds))
            # Change background of the modal to black
            st.markdown(
                """
                <style>
                .ant-modal-content {
                    background-color: #000000;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        color: white;
    }

    .css-18e3th9 {1

        color: black;
    }
    .css-1offfwp {
        color: white;
    }
    .st-b5rrxn {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    if "use_agent" not in st.session_state:
        st.session_state["use_agent"] = False

    if "selected_tools" not in st.session_state:
        st.session_state["selected_tools"] = []

    if "tool_input" not in st.session_state:
        st.session_state["tool_input"] = {}

    # Sidebar with tabs
    with st.sidebar:
        # tab1, tab2 = st.tabs(["Dialogue Settings", "Model Settings"])
        # tab1 = st.expander("Dialogue Settings", expanded=True)
        # tab1 = st.tabs(["Dialogue Settings"])
        # with tab1:
        # use_agent checkbox
        # st.session_state["use_agent"] = st.checkbox(
        #     "Enable Agent", st.session_state["use_agent"], help="Make sure the selected model has Agent capability"
        # )
        st.session_state["use_agent"] = False
        # Fetch tools from the API
        tools = api.list_tools()

        # Select tools based on use_agent state
        # if st.session_state["use_agent"]:
        #     st.session_state["selected_tools"] = st.multiselect(
        #         "Select Tools", list(tools), format_func=lambda x: tools[x]["title"],
        #         default=st.session_state["selected_tools"]
        #     )
        # else:
        #     selected_tool = st.selectbox(
        #         "Select Tool", list(tools), format_func=lambda x: tools[x]["title"],
        #         index=list(tools).index(st.session_state["selected_tools"][0]) if st.session_state["selected_tools"] else 0
        #     )
        #     print(f'selected_tool: {selected_tool}')
        st.session_state["selected_tools"] = ['search_local_knowledgebase']

        selected_tool_configs = {name: tool["config"] for name, tool in tools.items() if name in st.session_state["selected_tools"]}

        # When not using Agent, manually generate tool parameters
        if not st.session_state["use_agent"] and len(st.session_state["selected_tools"]) == 1:
            selected_tool = st.session_state["selected_tools"][0]
            # with st.expander("Tool Parameters", True):
            for k, v in tools[selected_tool]["args"].items():
                if choices := v.get("choices", v.get("enum")):
                    if st.session_state["tool_input"].get(k) is None:
                        st.session_state["tool_input"][k] = st.selectbox(v["title"], choices, key=k, index=0)
                    else:
                        st.session_state["tool_input"][k] = st.selectbox(v["title"], choices, key=k, index=choices.index(st.session_state["tool_input"].get(k, choices[0])))
                else:
                    if v["type"] == "integer":
                        st.session_state["tool_input"][k] = st.slider(v["title"], value=st.session_state["tool_input"].get(k, v.get("default")), key=k)
                    elif v["type"] == "number":
                        st.session_state["tool_input"][k] = st.slider(v["title"], value=st.session_state["tool_input"].get(k, v.get("default")), step=0.1, key=k)
                    else:
                        if selected_tool != "search_local_knowledgebase":    
                            st.session_state["tool_input"][k] = st.text_input(v["title"], v.get("default"), key=k)
                        else:
                            st.session_state["tool_input"][k] = None
        # uploaded_file = st.file_uploader("Upload Attachment", accept_multiple_files=False)
        # files_upload = process_files(files=[uploaded_file]) if uploaded_file else None
        files_upload = None
        # with tab2:
        # Conversation
        conv_names = list(st.session_state["conversation_ids"].keys())
        index = 0
        if st.session_state.get("cur_conv_name") in conv_names:
            index = conv_names.index(st.session_state.get("cur_conv_name"))
        # conversation_name = st.selectbox("Current Conversation", conv_names, index=index)
        conversation_name = conv_names[0]
        chat_box.use_chat_name(conversation_name)
        conversation_id = st.session_state["conversation_ids"][conversation_name]

        # Model
        # platforms = ["All"] + [x["platform_name"] for x in MODEL_PLATFORMS]
        # platform = st.selectbox("Select Model Platform", platforms)
        # llm_models = list(get_config_models(model_type="llm", platform_name=None if platform=="All" else platform))
        # llm_model = st.selectbox("Select LLM Model", llm_models)
        llm_model = 'gpt-4o'
        # Content to send to the backend
        chat_model_config = {key: {} for key in LLM_MODEL_CONFIG.keys()}
        for key in LLM_MODEL_CONFIG:
            if LLM_MODEL_CONFIG[key]:
                first_key = next(iter(LLM_MODEL_CONFIG[key]))
                chat_model_config[key][first_key] = LLM_MODEL_CONFIG[key][first_key]

        if llm_model is not None:
            chat_model_config['llm_model'][llm_model] = LLM_MODEL_CONFIG['llm_model'].get(llm_model, {})
            
    # Display chat messages from history on app rerun
    chat_box.output_messages()
    chat_input_placeholder = "Enter dialogue content, use Shift+Enter for line breaks. Type /help for custom commands."

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        if parse_command(text=prompt, modal=modal):
            st.rerun()
        else:
            history = get_messages_history(
                chat_model_config["llm_model"].get(next(iter(chat_model_config["llm_model"])), {}).get("history_len", 1)
            )
            chat_box.user_say(prompt)
            if files_upload:
                if files_upload["images"]:
                    st.markdown(f'<img src="data:image/jpeg;base64,{files_upload["images"][0]}" width="300">',
                                unsafe_allow_html=True)
                elif files_upload["videos"]:
                    st.markdown(
                        f'<video width="400" height="300" controls><source src="data:video/mp4;base64,{files_upload["videos"][0]}" type="video/mp4"></video>',
                        unsafe_allow_html=True)
                elif files_upload["audios"]:
                    st.markdown(
                        f'<audio controls><source src="data:audio/wav;base64,{files_upload["audios"][0]}" type="audio/wav"></audio>',
                        unsafe_allow_html=True)

            chat_box.ai_say("Thinking...")
            text = ""
            started = False

            client = openai.Client(base_url=f"{api_address()}/chat", api_key=MODEL_PLATFORMS[0]['api_key'])
            messages = history + [{"role": "user", "content": prompt}]
            tools = list(selected_tool_configs)
            if len(st.session_state["selected_tools"]) == 1:
                tool_choice = st.session_state["selected_tools"][0]
            else:
                tool_choice = None
            # If there are empty fields in tool_input, set them to user input
            for k in st.session_state["tool_input"]:
                if st.session_state["tool_input"][k] in [None, ""]:
                    st.session_state["tool_input"][k] = prompt

            extra_body = dict(
                            metadata=files_upload,
                            chat_model_config=chat_model_config,
                            conversation_id=conversation_id,
                            tool_input = st.session_state["tool_input"],
                            )
            chat_iterator = client.chat.completions.create(
                    messages=messages,
                    model=llm_model,
                    stream=True,
                    tools=tools,
                    tool_choice=tool_choice,
                    extra_body=extra_body,
                )
            
            for d in chat_iterator:
                message_id = d.message_id
                metadata = {
                    "message_id": message_id,
                }

                # Clear initial message
                if not started:
                    chat_box.update_msg("", streaming=False)
                    started = True

                content = process_content(d.choices[0].delta.content)

                if d.status == AgentStatus.error:
                    st.error(content)
                elif d.status == AgentStatus.llm_start:
                    chat_box.insert_msg("Analyzing tool output results...")
                    text = content
                elif d.status == AgentStatus.llm_new_token:
                    text += content
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=True, metadata=metadata)
                elif d.status == AgentStatus.llm_end:
                    text += content
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=False, metadata=metadata)
                elif d.status == AgentStatus.agent_finish:
                    text = content
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=False, metadata=metadata)
                elif d.status is None: # not agent chat
                    if getattr(d, "is_ref", False):
                        # Handle references specifically, escaping necessary characters
                        ref_content = process_content(d.choices[0].delta.content)
                        ref_content = format_markdown(ref_content)
                        chat_box.insert_msg(Markdown(ref_content, in_expander=True, state="complete", title="References"))
                        chat_box.insert_msg("")
                    else:
                        text += content
                        chat_box.update_msg(text.replace("\n", "\n\n"), streaming=True, metadata=metadata)

            chat_box.update_msg(text, streaming=False, metadata=metadata)

            if os.path.exists("tmp/image.jpg"):
                with open("tmp/image.jpg", "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                    img_tag = f'<img src="data:image/jpeg;base64,{encoded_string}" width="300">'
                    st.markdown(img_tag, unsafe_allow_html=True)
                os.remove("tmp/image.jpg")
            # chat_box.show_feedback(**feedback_kwargs,
            #                        key=message_id,
            #                        on_submit=on_feedback,
            #                        kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})

            # elif dialogue_mode == "文件对话":
            #     if st.session_state["file_chat_id"] is None:
            #         st.error("请先上传文件再进行对话")
            #         st.stop()
            #     chat_box.ai_say([
            #         f"正在查询文件 `{st.session_state['file_chat_id']}` ...",
            #         Markdown("...", in_expander=True, title="文件匹配结果", state="complete"),
            #     ])
            #     text = ""
            #     for d in api.file_chat(prompt,
            #                            knowledge_id=st.session_state["file_chat_id"],
            #                            top_k=kb_top_k,
            #                            score_threshold=score_threshold,
            #                            history=history,
            #                            model=llm_model,
            #                            prompt_name=prompt_template_name,
            #                            temperature=temperature):
            #         if error_msg := check_error_msg(d):
            #             st.error(error_msg)
            #         elif chunk := d.get("answer"):
            #             text += chunk
            #             chat_box.update_msg(text, element_index=0)
            #     chat_box.update_msg(text, element_index=0, streaming=False)
            #     chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    now = datetime.now()
    # Clear history button

    
    # with tab1:
    # cols = st.columns(2)
    # export_btn = cols[0]
    # if cols[1].button(
    #         "Clear History",
    #         use_container_width=True,
    # ):
    #     chat_box.reset_history()
    #     st.rerun()

    # warning_placeholder = st.empty()

    # export_btn.download_button(
    #     "Export Record",
    #     "".join(chat_box.export2md()),
    #     file_name=f"{now:%Y-%m-%d %H.%M}_Conversation_Record.md",
    #     mime="text/markdown",
    #     use_container_width=True,
    # )

    # st.write(chat_box.history)
