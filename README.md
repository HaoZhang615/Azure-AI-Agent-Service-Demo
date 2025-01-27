
# Azure AI Agent Service Demo

## Overview

Azure AI Agent Service Demo is a Streamlit-based web application that enables users to interact with an AI agent powered by Azure AI services. The application supports tools like Bing Grounding and Code Interpreter to enhance the AI's capabilities.

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
    - Create a `.env` file in the project root:
        ```env
        PROJECT_CONNECTION_STRING=your_connection_string
        BING_CONNECTION_NAME=your_bing_connection_name
        ```

## Usage

Run the Streamlit application:

```bash
streamlit run streamlit_frontend.py
```

Open the provided URL in your web browser to start interacting with the AI agent.

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