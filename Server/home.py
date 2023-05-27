from flask import Flask,request,render_template,url_for,redirect, session
from datetime import timedelta
from flask import flash
app = Flask(__name__,template_folder="/storage/emulated/0/Templates")
app.permanent_session_lifetime=timedelta(minutes=5)
app.secret_key="NIKHIL"
@app.route("/")
def home():
    return"<h1>Home Page</h1>"
@app.route("/login",methods=["POST","GET"])    
def login():
    if request.method=="GET":
        if "user" in session:
            value = session["user"]
            return f"<h2>You are already logged in as {value}!</h2>"
        else:
            return render_template("login.html")
    else:
        session.permanent=True
        user = request.form["name"]
        session["user"] = user
        return redirect(url_for("user"))
@app.route("/user")             
def user():
    if "user" in session: 
        value = session["user"]
        flash("successfully login","info by")
        return render_template("login_succ.html")
    else:
        flash("You are not logged in","info")
        return redirect(url_for("login"))
@app.route("/logout")
def logout():
    session.pop("user",None)
    flash("You have been logged out","info")
    return redirect(url_for("login"))
if __name__ =="__main__":
    app.run(debug = True)