# Heart Failure Prediction Flask Application 💔

This project implements a web application using Flask and a pre-trained machine learning model to predict the risk of heart failure based on various clinical parameters.

---

## Project Overview 📊

The goal of this project was to:
* Train a machine learning model (specifically, a **Gradient Boosting Classifier**) on the Heart Failure Clinical Records Dataset.
* Achieve an accuracy of over 80% for the trained model.
* Develop a Flask web application (`app.py`) that serves as a user interface for inputting patient data and receiving real-time predictions.
* Implement custom CSS (`index.html`) for a responsive and aesthetically pleasing user experience, following modern web design principles.

---

## Dataset & Model 🔬

* **Dataset**: The `heart_failure_clinical_records_dataset (1).csv` dataset was used for training and evaluating the model. It contains clinical data of patients and their `DEATH_EVENT` status (0 = No Death Event, 1 = Death Event) during follow-up.
* **Model**: A **Gradient Boosting Classifier** was chosen due to its robust performance in classification tasks. The trained model (`gradient_boosting_model.pkl`) and the StandardScaler (`scaler.pkl`) used for preprocessing are saved.

---

## How to Run Locally 🏃‍♀️

To set up and run this project on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/KnownTeach616/Heart-Failure-Prediction-Flask-App.git](https://github.com/KnownTeach616/Heart-Failure-Prediction-Flask-App.git)
    cd Heart-Failure-Prediction-Flask-App
    ```
2.  **Create and Activate a Virtual Environment:**
    * **Windows (Command Prompt):**
        ```bash
        python -m venv venv
        venv\Scripts\activate.bat
        ```
    * **Windows (PowerShell):**
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    * **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Flask Application:**
    ```bash
    flask run
    ```
5.  **Access the App:** Open your web browser and go to `http://127.0.0.1:5000/`.

---

## Application Screenshots 📸

Here are a few screenshots demonstrating the application's interface and functionality:

### Initial Form View:
![Initial Form View]({{ 'images/Screenshot_21-7-2025_153832_127.0.0.1.jpeg' | relative_url }})


### High-Risk Prediction Example:
![High-Risk Prediction]({{ 'images/Screenshot_21-7-2025_153935_127.0.0.1.jpeg' | relative_url }})


### Low-Risk Prediction Example:
![Low-Risk Prediction]({{ 'images/Screenshot_21-7-2025_154110_127.0.0.1.jpeg' | relative_url }})


---

## Files Included 📁

* `app.py`: The main Flask application.
* `Templates/index.html`: The HTML template for the web interface (with custom CSS).
* `gradient_boosting_model.pkl`: The trained machine learning model.
* `scaler.pkl`: The saved StandardScaler object for data preprocessing.
* `heart_failure_clinical_records_dataset (1).csv`: The dataset used for model training.
* `Heart_Failure_Prediction_Model_Training.ipynb`: Jupyter Notebook containing the model training and evaluation process.
* `requirements.txt`: Lists all Python dependencies.
* `.gitignore`: Specifies files and folders to be ignored by Git.
* `README.md`: This project documentation file.
* `images/`: Folder containing screenshots for this README.

---

## License 📜

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).