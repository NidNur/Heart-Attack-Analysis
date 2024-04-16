from flask import Flask, redirect, url_for, render_template, request, Blueprint

app = Flask(__name__)

@app.route("/")
def home_page(): 
    return render_template("home_page.html")

if __name__ == "__main__":
        app.run(debug=True)