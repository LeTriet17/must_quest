import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

# from chatchat.webui_pages.loom_view_client import update_store
# from chatchat.webui_pages.openai_plugins import openai_plugins_page
from chatchat.webui_pages.utils import *
from streamlit_option_menu import option_menu
from chatchat.webui_pages.dialogue.dialogue import dialogue_page, chat_box
from chatchat.webui_pages.knowledge_base.knowledge_base import knowledge_base_page
import os
import sys
from chatchat.configs import VERSION
from chatchat.server.utils import api_address

img_dir = os.path.dirname(os.path.abspath(__file__))
api = ApiRequest(base_url=api_address())

st.set_page_config(
        "Renn AI PDF Reader WebUI",
        os.path.join(img_dir, "img", "cq.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'About': f"""<p align="right">Current version {VERSION}</p>""",
        },
        layout="wide"
    )

# Initialize cookie manager
cookies = EncryptedCookieManager(
    prefix="renn_", 
    password="reNnlAbs@2024"
)
    

def login():
    st.session_state["logged_in"] = True
    cookies["logged_in"] = "True"  # Store as string
    cookies.save()

def logout():
    st.session_state["logged_in"] = False
    cookies["logged_in"] = "False"  # Store as string
    cookies.save()

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 325px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 400px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load cookies
    if not cookies.ready():
        st.stop()

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = cookies.get("logged_in", "False") == "True"  # Convert string back to bool

    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Chat"

    if st.session_state["logged_in"]:
        pages = {
            "Chat": {
                "icon": "chat",
                "func": dialogue_page,
            },
            "Knowledge Base": {
                "icon": "hdd-stack",
                "func": knowledge_base_page,
            },
        }

        with st.sidebar:
            st.image(os.path.join(img_dir, "img", "cq-long.png"), width=150)
            st.markdown(
                """
                <style>
                div.stImage > img {
                    margin-left: auto;
                    margin-right: auto;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <style>
                    [data-testid=stSidebar] [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            options = list(pages)
            icons = [x["icon"] for x in pages.values()]

            selected_page = option_menu(
                menu_title="",
                options=options,
                icons=icons,
                default_index=options.index(st.session_state["selected_page"]),
            )

            cols = st.columns(2)
            if cols[0].button(
                    "Clear History",
                    use_container_width=True,
            ):
                chat_box.reset_history()
                st.rerun()

            if cols[1].button("Logout"):
                logout()
                st.rerun()

        st.session_state["selected_page"] = selected_page

        if selected_page in pages:
            pages[selected_page]["func"](api=api, is_lite=is_lite)
    else:
        with st.form("login_form"):
            st.text_input("Username", key="username")
            st.text_input("Password", key="password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                # Add authentication logic here
                if st.session_state["username"] == "must" and st.session_state["password"] == "demo":
                    login()
                    st.rerun()
                else:
                    st.error("Invalid username or password")