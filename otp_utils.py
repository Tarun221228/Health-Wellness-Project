import random
import string

# Simple OTP storage (in production, use database or cache)
otp_storage = {}

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def store_otp(email, otp):
    otp_storage[email] = otp

def verify_otp(email, otp):
    stored_otp = otp_storage.get(email)
    if stored_otp and stored_otp == otp:
        del otp_storage[email]
        return True
    return False

def send_otp_email(email, otp):
    # For demo, print the OTP
    print(f"OTP for {email}: {otp}")
    # In production, send email
    # import smtplib
    # etc.
    return True
