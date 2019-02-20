# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import imageio


#################################################################################################
######################################## load video #############################################


def framecapture(filename):
    
#    cap = cv2.VideoCapture(videofilename)
#    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#    ret, frame = cap.read()
#    w = frame.shape[0]
#    h = frame.shape[1]
#    d = frame.shape[2]
#    print(framecount, w, h, d)
#    framearray = np.empty((framecount, w, h), dtype=frame.dtype)
#    i=0
#    
#    while(cap.isOpened()):
#        ret, frame = cap.read()
#        if ret==True:
#            #image processing here
#            _, frame = cap.read()
#            framearray[i]= frame[:, :, 0]
#            i=i+1
#            print(i)
#            #img = cv2.flip(frame,1)   # flip left-right  
#            #img = cv2.flip(img,0)     # flip up-down
#            #cv2.imshow('frame',frame)                                              # show the video
#            if i == 876:
#                break
#        else:
#            break
#    cap.release()
#    cv2.destroyAllWindows()
#    print("------------Video loaded------------------")
    
    videofilename = filename.replace(".",".avi")
    tsvfilename = filename.replace(".",".tsv")
    
   
    df = pd.read_table(tsvfilename, encoding = "ISO-8859-1")
    df = df.drop(0)
    
    cap = imageio.get_reader(videofilename)
    nr_of_images = cap.get_length()
    pix = 0.34
    
    Images = []
    for idx in range(nr_of_images):
        cellimg = cap.get_data(idx)
        cellimg = cellimg[:,:,0]
        Images.append(cellimg)
        #Images = np.array(Images)
        



    #read_csv('file', encoding = "ISO-8859-1")
#    i = 2
#    img = Images[i]
#    pos_x = float(df["pos_x"].iloc[i])/pix
#    pos_y = float(df["pos_y"].iloc[i])/pix
#
#    plt.imshow(img)
#    plt.scatter(pos_x,pos_y,c='w')
    
    #return[framearray]
    Images.append(df)
    return[Images, df]






frames_unknown, labels_unknown = framecapture(r"D:\Germany\UMAP_repo\20180211_3_45_19_unknown.")
frames_basos_ex, labels_basos_ex = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_basos_ex.")
frames_Bcells, labels_Bcells = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_Bcells.")
frames_CD3neg_NK, labels_CD3neg_NK = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_CD3neg_NK.")
frames_CD3pos_NK, labels_CD3pos_NK = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_CD3pos_NK.")
frames_debris, labels_debris = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_debris.")
frames_eos, labels_eos = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_eos.")
frames_ery, labels_ery = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_ery.")
frames_erydub, labels_erydub = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_erydub.")
frames_lymphos_ex, labels_lymphos_ex = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_lymphos_ex.")
frames_monos, labels_monos = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_mono.")
frames_neutro, labels_neutro = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_neutro.")
frames_Tcells, labels_Tcells = framecapture(r"D:\Germany\UMAP_repo\20180211_3_56_19_Tcells.")





#frames_Bcells = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_Bcells.avi")


#vfname = (r"D:\Germany\UMAP_repo\20180211_3_56_19-M1.avi")
#loaded_frames = framecapture(videofilename = vfname)


#frames_unknown = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_45_19_unknown.avi")
#frames_basos_ex = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_basos_ex.avi")
#frames_Bcells = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_Bcells.avi")
#frames_CD3neg_NK = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_CD3neg_NK.avi")
#frames_CD3pos_NK = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_CD3pos_NK.avi")
#frames_debris = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_debris.avi")
#frames_eos = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_eos.avi")
#frames_ery = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_ery.avi")
#frames_erydub = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_erydub.avi")
#frames_lymphos_ex = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_lymphos_ex.avi")
#frames_monos = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_mono.avi")
#frames_neutro = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_neutro.avi")
#frames_Tcells = framecapture(videofilename = r"D:\Germany\UMAP_repo\20180211_3_56_19_Tcells.avi")



px_res = 0.34
test_df = pd.read_csv(r'D:\Germany\UMAP_repo\20180211_3_56_19_Bcells.tsv', delimiter="\t", encoding='unicode_escape')
test_df2 = test_df['pos_x']
test_df2b = test_df2[1:]
test_df2bb = test_df2b.astype('float64')
print(test_df2bb.dtype)
test_df2c = test_df2bb/px_res


Bcells=frames_Bcells[0]
#show some images (set to RBC)
fig, ax_array = plt.subplots(3, 3)
axes = ax_array.flatten()
for i, ax in enumerate(axes):
    ax.imshow(Bcells[i,:,:], cmap='gray_r')
plt.setp(axes, xticks=[], yticks=[], frame_on=False)
plt.tight_layout(h_pad=0.01, w_pad=0.01)
print("------------Images shown------------------")


test_df3 = test_df['pos_y']


#
#plt.figure()
#plt.imshow(Bcells[0,:,:])
#plt.figure()
#plt.imshow(Bcells[1,:,:])
#plt.figure()
#plt.imshow(Bcells[2,:,:])
#plt.figure()
#plt.imshow(Bcells[3,:,:])
#plt.figure()
#plt.imshow(Bcells[4,:,:])
#plt.figure()
#plt.imshow(Bcells[5,:,:])







#################################################################################################
################################### load dataframes #############################################

fname = (r"D:\Germany\Blood-Data from Maik\001_DataSet\Lym_Eos_Mono_Neut_BGran_RBC_Deb_20180904_DataFrame_Train.csv")

#function to load excel file with parameters and .npy dataframe with images
def loadparams(csvfilename):
    "This will load parameters from the excel file"
    DFname = csvfilename
    DF = pd.read_csv(DFname, index_col=0)
    Xname = csvfilename.replace("DataFrame_Train.csv","Images_Train.npy")
    X = np.load(Xname)
    print("------------Data loaded------------------")
    return[X,DF]


blood_image, blood_df_parameters = loadparams(csvfilename = fname)
blood_image_cropped = blood_image[:, 12:36, :, :]                             #crops the image
blood_image_cropped1 = blood_image[1::3, 12:36, :, :]
blood_image_cropped2 = blood_image[2::3, 12:36, :, :]                           #takes every 3rd line  
blood_image_new = np.squeeze(blood_image_cropped1)                              #remove single dimension
blood_image_new2 = np.squeeze(blood_image_cropped2)        

blood_labels = blood_df_parameters['Label'].map({0:'lym', 1:'eos', 2:'mono', 3:'neut', 4:'bgran', 5:'rbc', 6:'deb'})
blood_labels1 = blood_labels[1::3]                                               # corresponding labels if every 3rd line is taken
blood_labels2 = blood_labels[2::3]  

blood_labels_numbers = blood_df_parameters['Label']
blood_labels_numbers1 = blood_labels_numbers[1::3]                               # only turn on when we want every 3rd event
blood_labels_numbers2 = blood_labels_numbers[2::3]                               # other third of data

#I want to create an array where 1 row = 1 image
#first allocate matrix and then fill with reshaped array
num_examples = blood_image_new.shape[0]
num_px = blood_image_new.shape[1]*blood_image_new.shape[2]
blood_image_new_flat=np.zeros((num_examples,num_px))
blood_image_new_flat= np.array(blood_image_new/255).reshape(num_examples,num_px)

num_examples2 = blood_image_new2.shape[0]
num_px2 = blood_image_new2.shape[1]*blood_image_new.shape[2]
blood_image_new_flat2= np.zeros((num_examples2,num_px2))
blood_image_new_flat2= np.array(blood_image_new2/255).reshape(num_examples2,num_px2)

#blood_image_new_flat= np.array(blood_image_new).flatten()
print("------------Array flattened------------------")


#show some images (set to RBC)
fig, ax_array = plt.subplots(5, 8)
axes = ax_array.flatten()
for i, ax in enumerate(axes):
    ax.imshow(blood_image_new[i+2100], cmap='gray_r')
plt.setp(axes, xticks=[], yticks=[], frame_on=False)
plt.tight_layout(h_pad=0.01, w_pad=0.01)
print("------------Images shown------------------")


#UMAP embedding
import umap
reducer = umap.UMAP(n_neighbors=15)
print("------------UMAP imported------------------")


#################################################################################################
################################### first embedding #############################################
#
##embed and time
#import time
#start = time. time()
#embedding = reducer.fit_transform(blood_image_new_flat, y=blood_labels_numbers1)
##embedding = reducer.fit_transform(blood_image_new_flat)
#embedding.shape
#end = time. time()
#print("------------UMAP embedding finished, embedding time = ------------------")
#print(end - start)
#
##scatter plot of embedding
##sns.scatterplot(embedding[:, 0], embedding[:, 1], hue = blood_labels2, marker='o', size=1)
#
###############################################################################################
#
##mouseover tooltips of images
#from io import BytesIO
#from PIL import Image
#import base64
#def embeddable_image(data):
#    img_data = data.astype(np.uint8)
#    #image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
#    image = Image.fromarray(img_data, mode='L')
#    buffer = BytesIO()
#    image.save(buffer, format='png')
#    for_encoding = buffer.getvalue()
#    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
#from bokeh.plotting import figure, show, output_file
#from bokeh.models import ColumnDataSource, CategoricalColorMapper
#from bokeh.palettes import Set1
#
#bloodembedding_df = pd.DataFrame(embedding, columns=('x', 'y'))
#bloodembedding_df['digit'] = [str(x) for x in blood_labels_numbers1]
#bloodembedding_df['image'] = list(map(embeddable_image, blood_image_new))
#
#
#output_file("learned_embedding.html")
#datasource = ColumnDataSource(bloodembedding_df)
#color_mapping = CategoricalColorMapper(factors=['0', '1', '2', '3', '4', '5', '6'],
#                                       palette=Set1[7])
#
##factors=['lym', 'eos', 'mono', 'neut', 'bgran', 'rbc', 'debris']
#
#tooltips1="""
#<div>
#    <div>
#        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
#    </div>
#    <div>
#        <span style='font-size: 16px; color: #224499'>Label:</span>
#        <span style='font-size: 18px'>@digit</span>
#    </div>
#</div>
#"""
#
#
#plot_figure = figure(
#    title='UMAP projection of the dataset',
#    plot_width=600,
#    plot_height=600,
#    tools=('pan, wheel_zoom, reset'),
#    tooltips = tooltips1
#    )
#
#
#
#plot_figure.circle(
#    'x',
#    'y',
#    source=datasource,
#    color=dict(field='digit', transform=color_mapping),
#    line_alpha=0.6,
#    fill_alpha=0.6,
#    size=4
#    )
#
#show(plot_figure)
#
#bloodembedding_df_1=bloodembedding_df

##############################################################################################
################################## first embedding #############################################

#embed and time
import time
start = time. time()
embedding = reducer.fit(blood_image_new_flat, blood_labels_numbers1)
#embedding = reducer.fit_transform(blood_image_new_flat)
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)

#scatter plot of embedding
#sns.scatterplot(embedding[:, 0], embedding[:, 1], hue = blood_labels2, marker='o', size=1)

##############################################################################################

#mouseover tooltips of images
from io import BytesIO
from PIL import Image
import base64
def embeddable_image(data):
    img_data = data.astype(np.uint8)
    #image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
    image = Image.fromarray(img_data, mode='L')
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Set1

bloodembedding_df = pd.DataFrame(embedding.embedding_, columns=('x', 'y'))
bloodembedding_df['digit'] = [str(x) for x in blood_labels_numbers1]
bloodembedding_df['image'] = list(map(embeddable_image, blood_image_new))


output_file("learned_embedding.html")
datasource = ColumnDataSource(bloodembedding_df)
color_mapping = CategoricalColorMapper(factors=['0', '1', '2', '3', '4', '5', '6'],
                                       palette=Set1[7])

#factors=['lym', 'eos', 'mono', 'neut', 'bgran', 'rbc', 'debris']

tooltips1="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Label:</span>
        <span style='font-size: 18px'>@digit</span>
    </div>
</div>
"""


plot_figure = figure(
    title='UMAP projection of the dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset'),
    tooltips = tooltips1
    )



plot_figure.circle(
    'x',
    'y',
    source=datasource,
    color=dict(field='digit', transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
    )

show(plot_figure)

bloodembedding_df_1=bloodembedding_df

##############################################################################################
################################## new embedding #############################################

#embed and time
import time
start = time. time()
test_embedding = embedding.transform(blood_image_new_flat2)
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)


##############################################################################################

bloodembedding_df_test = pd.DataFrame(test_embedding, columns=('x', 'y'))
bloodembedding_df_test['digit'] = [str(x) for x in blood_labels_numbers2]
bloodembedding_df_test['image'] = list(map(embeddable_image, blood_image_new2))


output_file("newly_embedded_points.html")
datasource2 = ColumnDataSource(bloodembedding_df_test)
color_mapping = CategoricalColorMapper(factors=['0', '1', '2', '3', '4', '5', '6'],
                                       palette=Set1[7])

tooltips1="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Label:</span>
        <span style='font-size: 18px'>@digit</span>
    </div>
</div>
"""


plot_figure = figure(
    title='UMAP projection of the dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset'),
    tooltips = tooltips1
    )


plot_figure.circle(
    'x',
    'y',
    source=datasource2,
    color=dict(field='digit', transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
    )

show(plot_figure)

bloodembedding_df_2=bloodembedding_df



##############################################################################################
################################## new embedding - 1 point #############################################

test=blood_image_new_flat2[0:1,:]
test_embedding2 = np.zeros((1,2))
#embed and time
import time
start = time. time()
test_embedding2 = embedding.transform(test)
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)
