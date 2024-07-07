from flask import Flask,render_template,jsonify, request ,redirect, url_for
import instaloader
import requests
import os
import random
import shutil 
import pickle
import praw 
import urllib.parse as _parse
import json
from pathlib import Path
import re # regex to detect username, url, html entity 
import nltk # to use word tokenize (split the sentence into words)
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model 

from transformers import IdeficsForVisionText2Text , AutoProcessor ,BitsAndBytesConfig
import torch

import easyocr
from torch.cuda.amp import autocast
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Adjust this path as needed



# Initialize the reader
reader = easyocr.Reader(['en'])

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    
    
    
    llm_int8_skip_modules=[
        "embed_tokens", "lm_head"
    ]
)
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_checkpoint = "HuggingFaceM4/idefics-9b"

image_processor = AutoProcessor.from_pretrained(base_model_checkpoint)

# Load the fine-tuned model and apply the adapter weights
model_checkpoint = "idefics-9b-memes-runpod/checkpoint-250"
image_model = IdeficsForVisionText2Text.from_pretrained(model_checkpoint, quantization_config=bnb_config )




comments_model = load_model('best_model.keras', compile=False)
comments_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize the tokenizer with appropriate settings
comments_tokenizer = Tokenizer(num_words=10000)

with open('tokenizer.pkl', 'rb') as handle:
    c_tokenizer = pickle.load(handle)

def _create_reddit_client():
    client = praw.Reddit(
        client_id = os.environ["client_id"],
        client_secret = os.environ["client_secret"],
        user_agent = os.environ["user_agent"]
    )
    return client

def _is_image(post):
    try:
        return post.post_hint == "image"
    except AttributeError:
        return False


def _get_img_urls(client: praw.Reddit, subreddit_name: str, limit: int):
    hot_memes = client.subreddit(subreddit_name).hot(limit = limit)
    image_urls = list()
    for post in hot_memes:
        # image_urls.append("https://reddit.com" + post.permalink +"#lightbox")
        if _is_image(post):
            image_urls.append(post.url)
    return image_urls

def _get_image_name(image_url: str) -> str:
    image_name = _parse.urlparse(image_url)
    return os.path.basename(image_name.path)

def _create_folder(folder_name: str):
    """
    If the folder does not exist then create the folder using the given name
    """
    try:
        os.mkdir(folder_name)
    except OSError:
        print("Something Happend")
    else:
        print("Folder Created")

def _download_image(folder_name: str, raw_response, image_name: str):
    _create_folder(folder_name)
    with open(f"{folder_name}/{image_name}", "wb") as image_file:
        shutil.copyfileobj(raw_response, image_file)

def _collect_memes(subreddit_name: str, limit: int=20):
    """
    Collects the images from the urls and stores them into the folders named after their subreddits
    """
    client = _create_reddit_client()
    images_urls = _get_img_urls(client=client, subreddit_name=subreddit_name, limit=limit)
    for image_url in images_urls:
        image_name = _get_image_name(image_url)
        response = requests.get(image_url, stream=True)

        if response.status_code == 200:
            response.raw.decode_content = True
            _download_image(subreddit_name, response.raw, image_name)





def extract_comments_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    comments = [post['text'] for post in data]
    return comments


def remove_noise_symbols(comment):
    text = comment.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')

    return text

def preprocess(comments):
    stop_words = set(stopwords.words('english'))
    clean_comments = []
    for comment in comments:
        # Remove URLs, HTML tags, and special characters
        comment = re.sub(r"http\S+|www\S+|https\S+", '', comment, flags=re.MULTILINE)
        comment = re.sub(r'@([^ ]+)', "user", comment)
        comment = re.sub(r'\<.*?\>', '', comment)
        comment = re.sub(r'[^\w\s]', '', comment)  # Only keep alphanumeric characters and spaces
        
        comment = remove_noise_symbols(comment)
        # Tokenize and remove stop words
        tokens = nltk.word_tokenize(comment)
        tokens = [word for word in tokens if word.lower() not in stop_words]

        cleaned_comment = " ".join(tokens)
        if cleaned_comment:  # Only add non-empty comments
            clean_comments.append(cleaned_comment)
    return clean_comments


def do_inference(model, processor, prompts, max_new_tokens=50):
    img_tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_tokens_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = img_tokenizer(bad_words, add_special_tokens=False).input_ids
    eos_token = "</s>"
    eos_token_id = img_tokenizer.convert_tokens_to_ids(eos_token)
    inputs = processor(prompts, return_tensors="pt").to(device)
    
    with autocast():
        generated_ids = model.generate(
            **inputs,
            eos_token_id=[eos_token_id],
            bad_words_ids=bad_words_ids,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def load_images_and_create_prompts(image_folder):
    prompts = []
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png')):
            
            image_path = image_folder +"/"+ filename
            
            image = cv2.imread(image_path)
            caption = ""
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform OCR
            result = reader.readtext(image_rgb)

            # Extract and print text
            for (bbox, text, prob) in result:
                caption += text + " "
            print(f"filename is{filename} and caption is{caption} \n")
                
            prompt = [image_path, f"Classify the image into a single category: Hateful or not hateful? Text on image:'{caption}'. Answer: "]
            prompts.append(prompt)
    
    
    return prompts





@app.route('/',methods=['POST','GET'])   
def index():

   
    if request.method == 'POST':
        username = request.form['username']
        appname = request.form['appname']
        
        if not username:
                return jsonify({'error': 'Username is required'}), 400
        if appname == "instagram":

            try:
                L = instaloader.Instaloader(download_pictures=True,save_metadata=False, download_comments= True, download_videos=False, download_video_thumbnails = False)
                L.login("kunal_.2305","sushma@23")
                L.post_metadata_txt_pattern = ""
                L.download_geotags = False
                
                L.save_metadata_json = False
        
                profile = instaloader.Profile.from_username(L.context, username)
                

                # Download the profile picture
                #L.download_profilepic(profile)

                posts = profile.get_posts()
                

                # # Create a folder to save the posts

                base_folder = 'static'

                image_folder = "static" + "/" + username + "_posts"
                # Save the current working directory
                original_cwd = os.getcwd()

                try:
                    # Change the working directory to base_folder
                    os.chdir(base_folder)
                    print(f"Changed working directory to: {base_folder}")
                    # Construct the full path for the new folder
                    folder_name = f'{username}_posts'

                    # Create the directory if it doesn't exist
                    os.makedirs(folder_name, exist_ok=True)
                    print(f"Folder created: {folder_name}")
                     # Filter and get only image posts
                    image_posts = [post for post in posts if post.typename == 'GraphImage' ]
            
                    num_posts_to_download = min(6, len(image_posts))
                    # Get random image posts
                    random_image_posts = random.sample(image_posts, num_posts_to_download)

                    predictions = []
                    for post in random_image_posts:
                    #comments = []
                        L.download_post(post, target=folder_name)
                    print("Image posts downloaded successfully.")

                finally:
                    # Change back to the original working directory
                    os.chdir(original_cwd)
                    print(f"Changed working directory back to: {original_cwd}")
                

                #profile_pic_filename = f'{username}_profile_pic.jpg'
                #shutil.move(profile_pic_filename, p / profile_pic_filename)

        
                

                prompts = load_images_and_create_prompts(image_folder=image_folder)
                print(f"Prompts created")
                
                predictions = []
                for prompt in prompts:
                   
                    prediction = do_inference(image_model, image_processor, prompt)
                    answer = prediction.split("Answer:")[-1].strip()
                     # Get the filename from the image path
                    filename = os.path.basename(prompt[0])
                    predictions.append({
                        'prompt': prompt,
                        'prediction': answer,
                        'image_path': username + '_posts' + '/' + filename,
                    })


                # all_comments=[]
                # for filename in os.listdir(folder_name):
                #     if filename.endswith(".json"):
                #         json_file = os.path.join(folder_name, filename)
                #         comments = extract_comments_from_json(json_file)
                #         all_comments.extend(comments)

                #         # Preprocess all comments and fit tokenizer
                # all_clean_comments = preprocess(all_comments)
                # comments_tokenizer.fit_on_texts(all_clean_comments)

                
                # # Process each JSON file for prediction
                # for filename in os.listdir(folder_name):
                #     if filename.endswith(".json"):
                #         json_file = os.path.join(folder_name, filename)
                #         comments = extract_comments_from_json(json_file)
                       
                #     try:
                #         # Preprocess each comment
                #         clean_comments = preprocess(comments)
                        

                #         c_predictions = []
                #         for clean_comment in clean_comments:
               
                #             # Tokenize and pad the comment for model prediction
                #             tokenized_comment = comments_tokenizer.texts_to_sequences([clean_comment])
                
                #             padded_comment = pad_sequences(tokenized_comment, maxlen=200)
                
                #             pred = comments_model.predict(padded_comment)
              
                #             pred_value = pred[0][0]  # Extract the scalar prediction value
                #             label = "Non-hateful" if pred_value < 0.5 else "Hateful"
                           

                #             c_predictions.append({
                #                 'comment': clean_comment,
                #                 'prediction': float(pred_value),
                #                 'label': label
                #                 })
                            
                       

                # except Exception as e:
                    #     return jsonify({'error': str(e)}), 500
                global stored_posts, name 
                name = username
                stored_posts = predictions
    
                return redirect(url_for('result'))



                
        
                
    
    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
                
            
        elif appname == "twitter":
            # Add your Twitter handling logic here
            return jsonify({'message': 'Twitter handling not implemented yet'}), 501

        elif appname == "reddit":
            try:
                _collect_memes(username)
                return jsonify({'message': 'Image posts downloaded successfully'}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        else:
            return jsonify({'error': 'Invalid appname'}), 400

    else:
        return render_template('index.html')

from werkzeug.utils import secure_filename

@app.route('/upload_meme', methods=['POST', 'GET'])
def upload_meme():
    if request.method == 'POST': 
        file = request.files['image']
    
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
    
        # Check if the upload folder is empty
        folder_empty = len(os.listdir(app.config['UPLOAD_FOLDER'])) == 0
    
        # Delete files only if the folder is not empty
        if not folder_empty:
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    if os.path.isfile(file_path) and filename != secure_filename(file.filename):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    
        # Save the new file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
    
        # Perform inference and get prediction
        prompt = load_images_and_create_prompts(image_folder=app.config['UPLOAD_FOLDER'])
        prediction = do_inference(image_model, image_processor, prompt)
        answer = prediction.split("Answer:")[-1].strip()
  
        # Prepare response
        response = {
            'prediction': answer,
            'image_path': url_for('static', filename='uploads/' + filename)  # Assuming 'uploads' is your upload folder under 'static'
        }
    
        # If Ajax POST request, return JSON response
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(response)
        else:
            # If not Ajax or GET request, render template with predictions
            return render_template('single.html', predictions=[response])
    
    # Render the upload form for GET request
    return render_template('single.html', predictions=[])
@app.route('/upload_tweet', methods=['POST', 'GET'])
def upload_tweet():
    if request.method == 'POST':
        tweet = request.form['tweet']
        tweet = str(tweet)
        tweets = [tweet]
        clean_text = preprocess(tweets)  # Assuming preprocess returns a list of cleaned strings
        
        t_predictions = []
        
        # Ensure clean_text is a list of strings
        if isinstance(clean_text, list) and len(clean_text) > 0:
            # Tokenize the input text
            tokenized_text = c_tokenizer.texts_to_sequences(clean_text)
            print("Tokenized text:", tokenized_text)  # Debugging output
            
            # Pad the tokenized sequences
            padded_text = pad_sequences(tokenized_text, maxlen=100)
            print("Padded text:", padded_text)  # Debugging output

            # Make predictions using the model
            pred = comments_model.predict(padded_text)
            pred_value = pred[0][0]  # Extract the scalar prediction value
            print("Prediction value:", pred_value) 

            # Determine the label based on the prediction value
            label = "Non-hateful" if pred_value > 0.5 else "Hateful"

            t_predictions.append({
                'comment': clean_text[0],  # Assuming only one tweet is processed
                'prediction': float(pred_value),
                'label': label
            })

        else:
            # Handle case where clean_text is empty or not a list
            print("Error: No valid input text found.")
        
        return jsonify(t_predictions)

    return render_template('single_tweet.html')

@app.route('/about_us') 
def about_us(): 
    return render_template('about_us.html') 
@app.route('/result')
def result():
    global stored_posts
    return render_template('result.html', predictions=stored_posts, username=name)

if __name__ == "__main__ ":
    app.run(debug=True)
   
    
    
    
    



