# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import imageio
import umap
import time
from io import BytesIO
from PIL import Image
import base64
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, CategoricalColorMapper, ContinuousColorMapper, LinearColorMapper
from bokeh.palettes import Set1, Paired, Oranges
def embeddable_image(data):                                                     #create png image for bokeh
    img_data = data.astype(np.uint8)
    #image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
    image = Image.fromarray(img_data, mode='L').resize((102,72))
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()

####################################################################################################
########################################### load video #############################################
####################################################################################################
    
def framecapture(filename):
       
    videofilename = filename.replace(".",".avi")
    tsvfilename = filename.replace(".",".tsv")
       
    df = pd.read_table(tsvfilename, encoding = "ISO-8859-1")
    df = df.drop(0)
    
    cap = imageio.get_reader(videofilename)
    nr_of_images = cap.get_length()
    pix = 0.34
    
    Images = []
    IMS = []
    
    for idx in range(nr_of_images):
        cellimg = cap.get_data(idx)
        cellimg = cellimg[:,:,0]
        
        pos_x = round(float(df["pos_x"].iloc[idx])/pix)
        pos_y = round(float(df["pos_y"].iloc[idx])/pix)
        #print(pos_x, pos_y)
        
        cellimg_cropped = cellimg[pos_y-12:pos_y+12,pos_x-17:pos_x+17]
        #cellimg_cropped = cellimg[pos_y-16:pos_y+16,pos_x-30:pos_x+30]
        cellimg_cropped_flattened = cellimg_cropped.flatten()
        cellimg_cropped_flattened = cellimg_cropped_flattened/255
        
        Images.append(cellimg_cropped_flattened)
        IMS.append(cellimg_cropped)
        
    plt.imshow(cellimg_cropped)
    #plt.scatter(pos_x,pos_y,c='w')
    Images=pd.DataFrame(Images)
    Table = df
    
    return[Images, Table]   
    
######################################################################################################
######################### embed new points using existing embedding ##################################
######################################################################################################
    
def new_embedding(frames_newembedding):
    
    frames_newembedding_removednan = np.nan_to_num(frames_newembedding)
    
    #embed and time
    start = time. time()
    embedding = embedding.transform(frames_newembedding_removednan)
    end = time. time()
    print("------------New UMAP embedding finished, embedding time = ", end - start, "------------------")

#    #if we want to embed only one point:
#    one_image_one_row=frames_newembedding[0:1,:] #takes first row from many rows where 1 row = 1 image
#    embedding = embedding.transform(one_image_one_row)

    num_im=frames_newembedding.shape[0]
    labels_newembedding_zeros = (np.zeros(num_im))
    reshaped_frames=(frames_newembedding.values.reshape(num_im,24,34))*255+60
    plt.imshow(reshaped_frames[0,:,:])
    
    newembedding_df = pd.DataFrame(embedding, columns=('x', 'y'))
    newembedding_df['digit'] = [str(x) for x in labels_newembedding_zeros]
    newembedding_df['image'] = list(map(embeddable_image, reshaped_frames))
    print("------------Dataframe with new embedding created------------------")

    return[newembedding_df] 


######################################################################################################
########################################## plot embedding ############################################
######################################################################################################

def plot_embedding(newembedding_df, color_mapping):

    output_file("UMAP_embedding.html")
    datasource = ColumnDataSource(newembedding_df)
    tooltips1="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 12px; color: #224499'>Label:</span>
                <span style='font-size: 12px'>@digit</span>
        </div>
        <div>
            <span style='font-size: 12px; color: #224499'> Index:</span>
                <span style='font-size: 12px'>@index</span>
        </div>             
    </div>
                """
    plot_figure = figure(
            title='UMAP projection of the dataset',
            plot_width=600,
            plot_height=600,
            tools=('pan, wheel_zoom, reset'),
            tooltips = tooltips1,
            #x_range=[-15, -7],
            #y_range=[-1, 9]
            x_range=[-22, 15],
            y_range=[-12, 20]
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
    
    return[]
  
###########################################################################################################
##################################         TRAINING DATASET         #######################################
###########################################################################################################
    
frames_basos_ex, table_basos_ex, = framecapture(r"E:\UMAP_repo\20180211_3_56_19_basos_ex.")
frames_Bcells, table_Bcells = framecapture(r"E:\UMAP_repo\20180211_3_56_19_Bcells.")
frames_CD3neg_NK, table_CD3neg_NK = framecapture(r"E:\UMAP_repo\20180211_3_56_19_CD3neg_NK.")
frames_CD3pos_NK, table_CD3pos_NK = framecapture(r"E:\UMAP_repo\20180211_3_56_19_CD3pos_NK.")
frames_debris, table_debris = framecapture(r"E:\UMAP_repo\20180211_3_56_19_debris.")
frames_eos, table_eos = framecapture(r"E:\UMAP_repo\20180211_3_56_19_eos.")
frames_ery, table_ery = framecapture(r"E:\UMAP_repo\20180211_3_56_19_ery.")
frames_erydub, table_erydub = framecapture(r"E:\UMAP_repo\20180211_3_56_19_erydub.")
frames_lymphos_ex, table_lymphos_ex = framecapture(r"E:\UMAP_repo\20180211_3_56_19_lymphos_ex.")
frames_monos, table_monos = framecapture(r"E:\UMAP_repo\20180211_3_56_19_mono.")
frames_neutro, table_neutro = framecapture(r"E:\UMAP_repo\20180211_3_56_19_neutro.")
frames_Tcells, table_Tcells = framecapture(r"E:\UMAP_repo\20180211_3_56_19_Tcells.")

# concatenate training images to 1 matrix, concatenate corresponding labels to vector with labels
frames = [frames_Bcells,frames_CD3neg_NK, frames_debris, frames_eos, frames_ery, frames_erydub, frames_monos, frames_neutro, frames_Tcells]
all_frames = pd.concat(frames)
all_frames_b= np.nan_to_num(all_frames)
all_labels=np.concatenate((np.zeros(frames_Bcells.shape[0]), np.zeros(frames_CD3neg_NK.shape[0])+1,
                           np.zeros(frames_debris.shape[0])+2, np.zeros(frames_eos.shape[0])+3, 
                           np.zeros(frames_ery.shape[0])+4, np.zeros(frames_erydub.shape[0])+5,
                           np.zeros(frames_monos.shape[0])+6, np.zeros(frames_neutro.shape[0])+7, 
                           np.zeros(frames_Tcells.shape[0])+8 ))

# train UMAP using training data and labels    
reducer = umap.UMAP(n_neighbors=15)
start = time. time()
embedding = reducer.fit(all_frames_b, all_labels)
#embedding = reducer.fit_transform(all_frames_b)
end = time. time()
print("------------UMAP embedding finished, embedding time = ", (end - start), "--------------")

# plot the embedding of the training dataset
reshaped_all=(all_frames.values.reshape(all_labels.shape[0],24,34))*255+60
plt.imshow(reshaped_all[0,:,:])

training_embedding_df = pd.DataFrame(embedding.embedding_, columns=('x', 'y'))
training_embedding_df['digit'] = [str(x) for x in all_labels]
training_embedding_df['image'] = list(map(embeddable_image, reshaped_all))
#print(bloodembedding_df['digit'].dtype)

colormapping_training = CategoricalColorMapper(factors=['0.0', '1.0', '2.0', '3.0', '4.0' '5.0', '6.0', '7.0', '8.0'],
                                       palette=Paired[9])
plot_embedding(training_embedding_df, colormapping_training)
print("------------Training embedding plotted-------------- ")


####################################################################################################################
############################################## new embedding - B-cells #############################################
####################################################################################################################

frames_newembedding, labels_newembedding, IMS_newembedding = framecapture(
        r"E:\UMAP_repo\20190208_CD3_CD56_CD19_Bcells2.") #load

newembedding_df = new_embedding(frames_newembedding)

color_mapping = CategoricalColorMapper(factors=['0.0'], palette=Oranges[3])
plot_embedding(newembedding_df, color_mapping)


