from flask import Blueprint


from flask import render_template
views = Blueprint('views',__name__)

x=10

@views.route('/', methods=["POST", "GET"])
@views.route('/home')
def homepage():
    return render_template("home.html")


