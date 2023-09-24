
# Creating a new Flask application and importing the necessary libraries, 
# such as NumPy and Scikit-learn (if you are using a pre-trained model)

from flask import Flask, render_template, request
import numpy as np
import pickle


# Create an instance of the Flask class and define routes for the web page.
app = Flask(__name__)
model=pickle.load(open('LR_model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

# Creating another route that will handle the form submission and use the model to make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    gre = float(request.form['gre'])
    gpa = float(request.form['gpa'])
    rank = int(request.form['rank'])
    feedback_text= request.form['feedback']

    features = np.array([gre, gpa, rank]).reshape(1, -1)
    probability = model.predict_proba(features)[0][1]
    admit=model.predict([[gre,gpa,rank]])
  
    if admit[0]==1:
        probability="Admitted"
    else :
        probability="Not Admitted"
     # Store input and output data in data.txt
    with open('data.txt', 'a') as file:
        file.write(f"GRE: {gre}, GPA: {gpa}, Rank: {rank}, Admission: {probability},Feedback: {feedback_text}\n")

    return render_template('predict.html', probability=probability)


# Finally, run the application using
if __name__ == '__main__':
    app.run(debug=True)
