import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["allnorth_consultants"]
usernames = ["rfocentral"]
passwords = ["116I90ZsU\-Z"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)
