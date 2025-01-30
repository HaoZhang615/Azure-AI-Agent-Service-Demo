# Azure AI Agents Demos

## Overview

Azure AI Agents Demos is a bundle of Streamlit-based demo level web application that enables users to interact with an AI agent powered by Azure AI services. The application supports tools like Bing Grounding and Code Interpreter to enhance the AI's capabilities.

## Features

- **Chat Interface:** Engage in conversations with the AI agent.
- **Bing Grounding:** Utilize Bing for enhanced information retrieval.
- **Code Interpreter:** Execute and interpret code snippets within the chat.
- **Configurable Settings:** Adjust AI parameters such as temperature and token limits.
- **Session Management:** Clear chat history and restart conversations as needed.

## Setup

### Prerequisites

- **Python 3.7+**
- **Azure Subscription:** Access to Azure AI Project services.
- **Git:** For cloning the repository.

### Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/Azure-AI-Agent-Service-Demo.git
    cd Azure-AI-Agent-Service-Demo
    ```

2. **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Environment Variables:**
    - Copy the `.env.example` file to `.env` and fill in your values:
        ```bash
        cp .env.example .env
        ```

## Usage

Run the Streamlit application:

```bash
streamlit run streamlit_frontend.py
```

Open the provided URL in your web browser to start interacting with the AI agent.

## UPDATE: Running the DeepSeek R1 Chatbot (requires DeepSeek R1 model deployed as Model as a Service in Azure AI Foundry, to obtain the endpoint and key) 
1. Make sure you have installed the required dependencies in your virtual environment.
2. Run the command:
   ```
   streamlit run streamlit_DeepSeek_R1.py
   ```
3. In your browser, use the sidebar to configure the temperature, max tokens, and system prompt (recommendation: Avoid adding a system prompt; all instructions should be contained within the user prompt).
4. Type your query in the chat input to interact with the model.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a New Branch:**
    ```bash
    git checkout -b feature/YourFeature
    ```
3. **Commit Your Changes:**
    ```bash
    git commit -m "Add your message"
    ```
4. **Push to the Branch:**
    ```bash
    git push origin feature/YourFeature
    ```
5. **Open a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.