from fastai.vision.all import (
    load_learner,
    PILImage,
)
import glob
import streamlit as st
from PIL import Image
from random import shuffle
import urllib.request

# set images
image1 = Image.open('sidebar1.jpg')
image2 = Image.open('menu1.jpg')
image3 = Image.open('menu2.jpg')

MODEL_URL = "https://github.com/GemmyTheGeek/BirdNerd/raw/main/BirdNerdV1.pkl"
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
learn_inf = load_learner('model.pkl', cpu=True)

thaimenu=[
"นกกระจอก","นกกระจิบ","นกกระจาบ"
]

engmenu=[
"Sparrow","Tailor Bird","Weaver Bird"
]

def predict(img, learn):
    # make prediction
    pred, pred_idx, pred_prob = learn.predict(img)
    # Display the prediction
    thaimenuname = thaimenu[int(pred_idx)]
    engmenuname = engmenu[int(pred_idx)]
    st.success(f"The type of bird is {engmenuname} ({thaimenuname}) with the probability of {pred_prob[pred_idx]*100:.02f}%")
    # Display the test image
    # st.image(img, use_column_width=True)
    col1.image(img, use_column_width=True)
    # col2.image(''+image3)
    col2.image(Image.open('./BirdCard/'+str(int(pred_idx))+'.png'))

##################################
# Top Main
##################################
col1, col2 = st.beta_columns(2)

##################################
# Col1
##################################
col1.header("Your Bird Image")

##################################
# Col2
##################################
col2.header("Information")

st.sidebar.image(image1)

fname = st.sidebar.file_uploader('Enter bird image to classify',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)
if fname is None:
    # fname = valid_images[0]
    col1.image(image2)
    col2.image(image3)
else :
    img = PILImage.create(fname)
    predict(img, learn_inf)

##################################
# sidebar
##################################
st.sidebar.markdown('This app, BirdNerd, was developed by Gemmy Somboontham and is a part of Bird Nerd Inc., organized by Bird Nerd Official')
st.sidebar.write("BirdNerd at Github [link](https://github.com/GemmyTheGeek/BirdNerd)")
st.sidebar.write("Checkout Colab [link](https://colab.research.google.com/github/GemmyTheGeek/BirdNerd/BirdnerdColab.ipynb)")
st.sidebar.write("Support Me! [link](https://www.youtube.com/c/codingforkids?sub_confirmation=1)")

st.text(' ')
st.text(' ')

my_expander = st.beta_expander(label='End Credits Scene')
with my_expander:
    'Gemmy : Thank You, Gemmy, for giving me more knowledge on Birds.'
    'Google Colab : Thank you, Google team, for making Colab; because of Colab, I was able to code more smoothly since I am used to it.'
    'DuckDuckGo : Thank you, DuckDuckGo, for giving me most of my dataset for Foody Dudy.'
    'Heroku : Thankyou to Heroku for letting me load my app in a blink of an eye, and being free.'
    'Family : I also would like to thankyou my family for giving me support during this time.'
    'Stackoverflow : I cant live without you.'
