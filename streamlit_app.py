"""
Streamlit UI for the Multi-Agent Analyst system.
"""

import os
import streamlit as st
import requests

# Configuration - Use environment variable or default to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Multi-Agent Analyst",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Multi-Agent Analyst")
st.markdown("Query the system using RAG, Database, or Forecasting agents.")

# Sidebar - Health Check
with st.sidebar:
    st.header("⚙️ System Status")
    
    if st.button("Check Health", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ {data['status'].upper()}")
                st.info(data['message'])
            else:
                st.error(f"❌ Unhealthy (Status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Is the server running?")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    st.markdown("### 💡 Example Queries")
    st.markdown("""
    **RAG (Knowledge):**
    - What are the error codes?
    - Explain the troubleshooting steps
    
    **Database:**
    - Show ticket count for last 3 days
    - Get statistics from 2026-01-25
    
    **Forecasting:**
    - Forecast tickets for next 5 days
    - Predict ticket volume for 7 days
    """)

# Main content
st.header("📝 Submit Query")

# Query input
query = st.text_area(
    "Enter your question:",
    placeholder="e.g., Forecast ticket count for next 3 days",
    height=100
)

# Submit button
col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("🚀 Submit", type="primary", use_container_width=True)
with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.rerun()

# Process query
if submit and query.strip():
    with st.spinner("Processing your query..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query},
                timeout=120  # 2 min timeout for complex queries
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display answer
                st.header("📤 Response")
                st.success(data['answer'])
                
                # Display work details in expander
                with st.expander("🔍 View Agent Work Details", expanded=False):
                    st.json(data['work'])
                
                # Show steps
                st.caption(f"Completed in {data['steps']} step(s)")
                
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"Error: {error_detail}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the server is running:")
            st.code("python main.py")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The query might be too complex.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif submit:
    st.warning("Please enter a query.")

# Footer
st.divider()
st.caption("Multi-Agent Analyst | RAG • Database • Forecasting")
