import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import time


# Fetch the service account key JSON file contents
cred = credentials.Certificate(
    'Insert credentials')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'Insert databaseURL'
})
ref = db.reference('output')