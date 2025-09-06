from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os, glob
import librosa
import librosa.display
import IPython
from IPython.display import Audio
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings('ignore')
import pickle
import soundfile
import sounddevice as sd
import soundfile as sf
from tkinter import ttk
from playsound import playsound
import multiprocessing

main = tkinter.Tk()
main.title("Song recommendation based on voice tone analysis") #designing main screen
main.geometry("800x700")

global filename, model, tf1, songs, player, textarea

class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self,num_emotions):
        super().__init__() 
        
        ################ TRANSFORMER BLOCK #############################
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        
       
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=0.4, 
            activation='relu' 
        )
        
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        
        self.conv2Dblock1 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout(p=0.3), 
            
           
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
           
            nn.Conv2d(
                in_channels=32,
                out_channels=64, 
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
       
        self.conv2Dblock2 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16),
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Dropout(p=0.3), 
            
           
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), 
            nn.Dropout(p=0.3), 
            
        
            nn.Conv2d(
                in_channels=32,
                out_channels=64, 
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )

        ################# FINAL LINEAR BLOCK ####################
      
        self.fc1_linear = nn.Linear(512*2+40,num_emotions) 
        
        ### Softmax layer for the 8 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1) # dim==1 is the freq embedding
        
    # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self,x):
        
        
        conv2d_embedding1 = self.conv2Dblock1(x) # x == N/batch * channel * freq * time
      
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1) 
        
        ############ 2nd parallel Conv2D block: 4 Convolutional layers #############################
        conv2d_embedding2 = self.conv2Dblock2(x) # x == N/batch * channel * freq * time
        
      
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1) 
        
        x_maxpool = self.transformer_maxpool(x)

        x_maxpool_reduced = torch.squeeze(x_maxpool,1)
        
      
        x = x_maxpool_reduced.permute(2,0,1) 
 
        transformer_output = self.transformer_encoder(x)
        
  
        transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x70 --> 40
        
   
        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2,transformer_embedding], dim=1)  

        output_logits = self.fc1_linear(complete_embedding)  
        
        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)
        

        return output_logits, output_softmax                       

emotions_dict ={
    '0':'surprised',
    '1':'neutral',
    '2':'calm',
    '3':'happy',
    '4':'sad',
    '5':'angry',
    '6':'fearful',
    '7':'disgust'
}
sample_rate = 48000
with open('scaler.txt', 'rb') as file:
    scaler = pickle.load(file)
file.close()
def feature_melspectrogram(waveform, sample_rate, fft = 1024, winlen = 512, window='hamming', hop=256, mels=128,):
    melspectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=fft, win_length=winlen, window=window, hop_length=hop, n_mels=mels, fmax=sample_rate/2)
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    return melspectrogram

def feature_mfcc(waveform, sample_rate, n_mfcc = 40, fft = 1024, winlen = 512, window='hamming', mels=128):
    mfc_coefficients = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc, n_fft=fft, win_length=winlen, window=window, n_mels=mels, fmax=sample_rate/2)
    return mfc_coefficients

def get_features(waveforms, features, samplerate):
    file_count = 0
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        file_count += 1
        print('\r'+f' Processed {file_count}/{len(waveforms)} waveforms',end='')
    return features

def get_waveforms(file):
    waveform, _ = librosa.load(file, duration=3, offset=0.5, sr=sample_rate)
    waveform_homo = np.zeros((int(sample_rate*3,)))
    waveform_homo[:len(waveform)] = waveform
    return waveform_homo

def loadModel():
    global model
    textarea.delete('1.0', END)
    model = parallel_all_you_want(len(emotions_dict))
    checkpoint_dict = torch.load('./models/parallel_all_you_wantFINAL-013.pkl')
    model.load_state_dict(checkpoint_dict['model'])
    textarea.insert(END,"Transformers Based Speech Emotion Detection Model loaded\n")    

def recommendSongs(emotion):
    global tf1, songs
    if emotion == 'calm' or emotion == 'happy' or emotion == 'neutral':
        names = ""
        for root, dirs, directory in os.walk('josh'):
            for j in range(len(directory)):
                names += 'josh/'+directory[j]+","
        names = names.strip()
        names = names[0:len(names)-1]
        names = names.split(",")
        songs = list(set([record for record in names]))
        print(songs)
        tf1['values'] = songs
        tf1.current(0)
    else:
        names = ""
        for root, dirs, directory in os.walk('happy'):
            for j in range(len(directory)):
                names += 'happy/'+directory[j]+","
        names = names.strip()
        names = names[0:len(names)-1]
        names = names.split(",")
        songs = list(set([record for record in names]))
        print(songs)
        tf1['values'] = songs
        tf1.current(0)

def demandRecommendation():
    global main
    textarea.delete('1.0', END)
    query = simpledialog.askstring("What type of songs you want Happy or Josh?","What type of songs you want Happy or Josh? Type happy or josh")
    query = query.lower().strip()
    if query == 'josh':
        query = 'happy'
    else:
        query = 'sad'
    recommendSongs(query)
    textarea.insert(END,"See Recommended Song names in Drop Down Box")

def stopPlaying():
    global player
    player.terminate()

def playSong():
    global player
    name = tf1.get()
    player = multiprocessing.Process(target=playsound, args=(name,))
    player.start()

def predictEmotion():
    textarea.delete('1.0', END)
    global filename, model, sample_rate
    textarea.insert(END,"Please start recording\n\n")
    textarea.update_idletasks()
    recording = sd.rec(int(3 * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float, blocking=True)
    sf.write('audio.wav', recording, sample_rate)
    textarea.insert(END,"Recording Completed\n\n")
    textarea.update_idletasks()
    waveform = get_waveforms('audio.wav')
    waveforms = []
    waveforms.append(waveform)
    waveforms = np.array(waveforms)
    print(waveforms.shape)
    testData = []
    testData = get_features(waveforms, testData, sample_rate)
    testData = np.asarray(testData)
    testData = np.reshape(testData, (testData.shape[0], (testData.shape[1] * testData.shape[2]))) 
    print(testData.shape)
    testData = scaler.transform(testData)
    testData = np.reshape(testData, (testData.shape[0], 40, 282))
    print(testData.shape)
    testData = np.expand_dims(testData,1)
    print(testData.shape)
    testData = torch.tensor(testData).float()
    output_logits, output_softmax = model(testData)
    predictions = torch.argmax(output_softmax,dim=1)
    predictions = predictions.numpy()
    predictions = predictions[0]
    print(predictions)
    if predictions > 0:
        emotion = emotions_dict.get(str(predictions - 1))
    else:
        emotion = emotions_dict.get(str(predictions))
    textarea.insert(END,"Emotion Predicted as : "+emotion)
    textarea.update_idletasks()
    recommendSongs(emotion)

def close():
    main.destroy()

def runGUI():
    global tf1, songs, textarea
    font = ('times', 16, 'bold')
    title = Label(main, text='Song recommendation based on voice tone analysis', justify=LEFT)
    title.config(bg='lavender blush', fg='DarkOrchid1')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=100,y=5)
    title.pack()

    font1 = ('times', 13, 'bold')
    model = Button(main, text="Generate & Load Voice Emotion Model", command=loadModel)
    model.place(x=200,y=100)
    model.config(font=font1)  

    predictButton = Button(main, text="Record & Predict Emotion", command=predictEmotion)
    predictButton.place(x=100,y=150)
    predictButton.config(font=font1)

    demandButton = Button(main, text="On Demand Recommendation", command=demandRecommendation)
    demandButton.place(x=360,y=150)
    demandButton.config(font=font1)

    l1 = Label(main, text='Recommended Songs')
    l1.config(font=font1)
    l1.place(x=100,y=200)

    songs = []
    songs.append("Recommended Songs")
    tf1 = ttk.Combobox(main,values=songs,postcommand=lambda: tf1.configure(values=songs))
    tf1.place(x=360,y=200)
    tf1.config(font=font1)
    
    playButton = Button(main, text="Play Song", command=playSong)
    playButton.place(x=100,y=250)
    playButton.config(font=font1)

    stopButton = Button(main, text="Stop Playing", command=stopPlaying)
    stopButton.place(x=240,y=250)
    stopButton.config(font=font1)

    font1 = ('times', 12, 'bold')
    textarea=Text(main,height=18,width=120)
    scroll=Scrollbar(textarea)
    textarea.configure(yscrollcommand=scroll.set)
    textarea.place(x=10,y=300)
    textarea.config(font=font1)

    main.config(bg='light coral')
    main.mainloop()



if __name__ == '__main__':
    runGUI()
