# Import necessary libraries
from flask import Flask

app = Flask(__name__)

# Import routes
from app import main
