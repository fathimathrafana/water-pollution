 **WATER POLLUTION DETECTION PROJECT 💧🔬**

 **OVERVIEW**
Water pollution is a growing global challenge, affecting health, ecosystems, and access to clean drinking water.  
This project uses Deep Learning to classify water images into two categories: **Clean Water** and **Dirty Water**.  
The goal is to support early detection of contaminated water sources, enable quick response, and promote safer water usage. 

**DATASET**
https://github.com/pooj180304/Water-quality-detection-using-deep-cnn-image-classifier/blob/main/water%20images.zip

- Total Classes:2  
- Classes: Clean-samples, Dirty-samples  
- Dataset is organized in folders by class and contains a Train/Test split.

**TECHNOLOGIES USED**
- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib**
- **OpenCV** (for image preprocessing)
- **Google Colab** (for model training)

- 
**📌 Features**
  
Upload an image of water and get a predicted category (Clean / Polluted).
Displays prediction confidence.
Built with Flask, TensorFlow Lite, and NumPy.
Frontend built using HTML, CSS, and JavaScript.
Deployed on Render (free tier)

**⚙️ Tech Stack**

Backend: Flask (Python)
Model: TensorFlow Lite (CNN model)
Frontend: HTML, CSS, JavaScript
Deployment: Render (Python Web Service)

**🚀 Installation & Setup**

**1.Clone the repository:**

git clone https://github.com/yourusername/water-pollution-detection.git
cd water-pollution-detection


**2.Create and activate a virtual environment:**

python3 -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows


**Install dependencies:**
pip install -r requirements.txt


**Run the Flask app:**
python app.py


**Open in browser:**
http://127.0.0.1:5000

**🌐 Deployment (Render)**

1.Push your project to GitHub.

2.On Render:

  -Create a new Web Service.

  -Connect your GitHub repo.

  -Build command:

     pip install -r requirements.txt
  
  -Start command:

     gunicorn app:app 
   
 📜 License

     This project is released under the MIT License.

👩‍💻 Author

Developed by Fathimath Rafana CR. Contributions are welcome! 🚀

🌍 Live Demo: https://water-pollution-13.onrender.com/


Here are some previews of the project:

![Screenshot 1](assets/screenshot1.png)
![Screenshot 2](assets/screenshot2.png)
