# Mobile Shop Chatbot

A Flask-based web application with an integrated AI chatbot for a mobile phone shop. The chatbot helps users query product prices, screen sizes, and memory configurations using LSTM-based natural language processing.

## Project Structure

The repository is organized as follows:

-   **`app.py`**: The main Flask application entry point.
-   **`chatbot_core.py`**: Core logic for the chatbot, handling user queries and database interactions.
-   **`model_loader.py`**: Utility to load the pre-trained Keras models (`chatbot.h5`, `classification.h5`).
-   **`models/`**: Contains the trained machine learning models.
    -   `chatbot.h5`: Sequence-to-sequence model for generating responses.
    -   `classification.h5`: Model for intent classification.
-   **`scripts/`**: Utility scripts for database management and maintenance.
-   **`templates/`**: HTML templates for the web interface.
-   **`static/`**: Static assets (CSS, JS, images).
-   **`instance/`**: Contains the SQLite database (`giasanpham.db`).

## Features

-   **Web Interface**: A clean e-commerce style interface showcasing mobile phones.
-   **AI Chatbot**: An embedded chatbot that can answer questions about:
    -   Product Prices ("What is the price of iPhone 15?")
    -   Screen Sizes ("Screen size of Samsung Galaxy?")
    -   Memory/Storage ("Memory of Oppo Find N2?")
-   **Database Integration**: Real-time product data retrieval from a SQLite database.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd DPL302m-Group5
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed. Install the required packages (create a `requirements.txt` if needed, common dependencies listed below):
    ```bash
    pip install flask pandas numpy matplotlib seaborn tensorflow scikit-learn nltk
    ```

3.  **Run the Application**:
    ```bash
    python app.py
    ```

4.  **Access the Website**:
    Open your browser and navigate to `http://127.0.0.1:5000/`.

## Usage

-   Browse the home page to see featured phones.
-   Click the "Chat" button in the bottom corner to open the chatbot.
-   Type questions like:
    -   "What is the price of iPhone 15 Pro Max?"
    -   "Does the shop have Samsung S23?"

## Development

-   **Scripts**: Use scripts in the `scripts/` directory to inspect or modify the database.
    -   `python scripts/inspect_db.py` to view current data.

## Credits

Developed by Group 5 for DPL302m.
