from flask import Flask,render_template,url_for,request
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__,template_folder="/storage/emulated/0/Templates/")

@app.route("/",methods=["POST","GET"])
def home():
    if request.method=="POST":
        sl = request.form['sepal_len']
        sw = request.form['sepal_wd']
        pl = request.form['petal_len']
        pw = request.form['petal_wd']
        rc = RandomForestClassifier()
        model = pickle.load(open("model.pkl","rb"))
        vals = [[float(sl),float(sw),float(pl),float(pw)]]
        total = model.predict(vals)
        return render_template("predict.html", content = total)
    else:     
        return render_template("iris.html")
    
if __name__ =="__main__":
     app.run(debug=True)