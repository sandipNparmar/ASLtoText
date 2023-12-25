import pyttsx3
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
#print(voices)
engine.setProperty('voice',voices[0].id)
engine.setProperty('rate',170)
def speak(text):
    engine.say(text=text)
    engine.runAndWait()