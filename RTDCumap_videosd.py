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
from bokeh.models import ColumnDataSource, CategoricalColorMapper, LinearColorMapper, ColorBar
from bokeh.palettes import Set1, Paired, Oranges, Viridis256
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
    
    Table = df.reset_index()
    Table = Table.drop('level_0', axis=1)  
    I_T_df = pd.concat([Images, Table], axis = 1)
    
    return[I_T_df, Images, Table]  
    
######################################################################################################
######################### embed new points using existing embedding ##################################
######################################################################################################
    
def new_embedding(trained_embedding, frames_newembedding):
    
    frames_newembedding_removednan = np.nan_to_num(frames_newembedding)
    
    #embed and time
    start = time. time()
    newembedding = trained_embedding.transform(frames_newembedding_removednan)
    end = time. time()
    print("------------New UMAP embedding finished, embedding time = ", end - start, "------------------")

#    #if we want to embed only one point:
#    one_image_one_row=frames_newembedding[0:1,:] #takes first row from many rows where 1 row = 1 image
#    embedding = embedding.transform(one_image_one_row)

    return[newembedding]

######################################################################################################
######################### Create DF with embedding, parameters, labels ###############################
######################################################################################################    

def create_df(all_embedded, all_frames, all_labels, all_tables):
    
    all_frames_reshaped=(all_frames.values.reshape(all_frames.shape[0],24,34))*255+60
    #plt.imshow(all_frames_reshaped[0,:,:])    
    em_df = pd.DataFrame(all_embedded, columns=('x', 'y'))
    em_df['digit'] = [str(x) for x in all_labels]
    em_df['image'] = list(map(embeddable_image, all_frames_reshaped))
    em_df = pd.concat([em_df, all_tables], axis=1)
    
    return[em_df]    

######################################################################################################
################################ PLOT embedding - training data ######################################
######################################################################################################

def plot_embedding(df_embedding, title):

    output_file("UMAP_embedding.html")
    df_embedding = df_embedding.drop('level_0', axis=1)
    datasource = ColumnDataSource(df_embedding)
    
    #mapper = CategoricalColorMapper(factors=['0', '1', '2', '3', '4' '5', '6', '7', '8'],
    #                                   palette=Paired[9])
    #mapper = CategoricalColorMapper(factors=['0.0'], palette=Oranges[3])
    
    mapper = LinearColorMapper(palette=Viridis256, low=0, high=0.1)
    color_bar = ColorBar(color_mapper=mapper, location=(0,0), label_standoff=8, major_label_text_font_size="10pt")
    
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
            title=title,
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
            color=dict(field='deform', transform=mapper),   ################### change to 'digit' if we want labels
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4
            )

    plot_figure.add_layout(color_bar, 'right')
    show(plot_figure)
    
    return[]

###########################################################################################################
##################################         TRAINING DATASET         #######################################
###########################################################################################################
    
df_basos_ex, frames_basos_ex, table_basos_ex = framecapture(r"E:\UMAP_repo\20180211_3_56_19_basos_ex.")
df_Bcells, frames_Bcells, table_Bcells = framecapture(r"E:\UMAP_repo\20180211_3_56_19_Bcells.")
df_CD3neg_NK, frames_CD3neg_NK, table_CD3neg_NK = framecapture(r"E:\UMAP_repo\20180211_3_56_19_CD3neg_NK.")
df_CD3pos_NK, frames_CD3pos_NK, table_CD3pos_NK = framecapture(r"E:\UMAP_repo\20180211_3_56_19_CD3pos_NK.")
df_debris, frames_debris, table_debris = framecapture(r"E:\UMAP_repo\20180211_3_56_19_debris.")
df_eos, frames_eos, table_eos = framecapture(r"E:\UMAP_repo\20180211_3_56_19_eos.")
df_ery, frames_ery, table_ery = framecapture(r"E:\UMAP_repo\20180211_3_56_19_ery.")
df_erydub, frames_erydub, table_erydub = framecapture(r"E:\UMAP_repo\20180211_3_56_19_erydub.")
df_lymphos_ex, frames_lymphos_ex, table_lymphos_ex = framecapture(r"E:\UMAP_repo\20180211_3_56_19_lymphos_ex.")
df_monos, frames_monos, table_monos = framecapture(r"E:\UMAP_repo\20180211_3_56_19_mono.")
df_neutro, frames_neutro, table_neutro = framecapture(r"E:\UMAP_repo\20180211_3_56_19_neutro.")
df_Tcells, frames_Tcells, table_Tcells = framecapture(r"E:\UMAP_repo\20180211_3_56_19_Tcells.")
print("------------ Training images loaded--------------")

# concatenate training images to 1 matrix, concatenate corresponding labels to vector with labels
frames = [frames_Bcells, frames_CD3neg_NK, frames_debris, frames_eos, frames_ery, frames_erydub, frames_monos, frames_neutro, frames_Tcells]
all_frames = pd.concat(frames)
all_frames_b = np.nan_to_num(all_frames)

#create and concatenate labels
all_labels= np.concatenate((np.zeros(frames_Bcells.shape[0]), np.zeros(frames_CD3neg_NK.shape[0])+1,
                           np.zeros(frames_debris.shape[0])+2, np.zeros(frames_eos.shape[0])+3, 
                           np.zeros(frames_ery.shape[0])+4, np.zeros(frames_erydub.shape[0])+5,
                           np.zeros(frames_monos.shape[0])+6, np.zeros(frames_neutro.shape[0])+7, 
                           np.zeros(frames_Tcells.shape[0])+8 ))
all_labels = all_labels.astype(int)
all_labels_df = pd.DataFrame(all_labels, columns=['labels'])

#concatenate tables
tables = [table_Bcells, table_CD3neg_NK, table_debris, table_eos, table_ery, table_erydub, table_monos, table_neutro, table_Tcells]
all_tables = pd.concat(tables)
all_tables = all_tables.reset_index()

##concatenate dataframes
#dataframes = [df_Bcells, df_CD3neg_NK, df_debris, df_eos, df_ery, df_erydub, df_monos, df_neutro, df_Tcells]
#all_df = pd.concat(dataframes)
#all_df = all_df.reset_index()
#all_df_labelled = pd.concat([all_df, all_labels_df], axis=1)
#all_df_labelled= np.nan_to_num(all_df_labelled)

###########################################
###########################################
# train UMAP using training data and labels    
reducer = umap.UMAP(n_neighbors=15)
print("------------ Embedding in progress --------------")
start = time. time()
embedding = reducer.fit(all_frames_b, all_labels)
#embedding = reducer.fit_transform(all_frames)
end = time. time()
print("------------UMAP embedding finished, embedding time = ", (end - start), "--------------")

###########################################
###########################################
#create dataframe of embedding
training_embedding_df = create_df(embedding.embedding_, all_frames, all_labels, all_tables)
training_embedding_df = training_embedding_df[0]

#plot embedding of training data
title_training = 'Embedding of training data'
plot_embedding(training_embedding_df, title_training)
print("------------Training embedding plotted-------------- ")


####################################################################################################################
#################################### embedding and plotting of new points ##########################################
####################################################################################################################

#load new file, create fake labels
df_new, frames_new, table_new = framecapture(
#        r"E:\UMAP_repo\20180211_3_45_19_unknown.") #load
        r"E:\UMAP_repo\20190208_CD3_CD56_CD19_Bcells2.") #load
labels_new = np.zeros(frames_new.shape[0])

#use trained embedding to embed new points (frames_new)
newembedding = new_embedding(embedding, frames_new)

#create dataframe with embedding, frames, labels (fake 0) and table with parameters
df_newembedding = create_df(newembedding, frames_new, labels_new, table_new)

#create plot of new embedding
title = 'Embedding of Unknown'
plot_embedding(df_newembedding, title)



