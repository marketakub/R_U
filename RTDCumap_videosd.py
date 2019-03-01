# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import imageio
import umap
import time
import pickle
from datetime import datetime
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
from bokeh.io import output_file, show
from bokeh.layouts import widgetbox, column, row, gridplot
from bokeh.models.widgets import Select, Button
from bokeh.models.callbacks import CustomJS

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
        
        #print(df["pos_x"])
        pos_x = round(float(df["pos_x"].iloc[idx])/pix)
        pos_y = round(float(df["pos_y"].iloc[idx])/pix)
        #print(pos_x, pos_y)
        
        cellimg_cropped = cellimg[pos_y-18:pos_y+18,pos_x-24:pos_x+24] 
        #cellimg_cropped = cellimg[pos_y-11:pos_y+11,pos_x-16:pos_x+14] #used for lymphocytes
        #cellimg_cropped = cellimg[pos_y-12:pos_y+12,pos_x-17:pos_x+17] #used for original embedding Feb 2019
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
    
    all_frames_reshaped=(all_frames.values.reshape(all_frames.shape[0],36,48))*255+60
    #all_frames_reshaped=(all_frames.values.reshape(all_frames.shape[0],24,34))*255+60
    #plt.imshow(all_frames_reshaped[0,:,:])    
    em_df = pd.DataFrame(all_embedded, columns=('x', 'y'))
    em_df['digit'] = [str(x) for x in all_labels]
#    em_df['digit2'] = em_df['digit'].map({'0':'lym_B', '1':'lym_NK', '2':'debris', '3':'eos', '4':'RBC', 
#         '5':'RBC_2', '6':'monos', '7':'neutro', '8':'lym_T', '100':'new'})
    em_df['digit2'] = em_df['digit'].map({'0':'lym_B', '1':'ECC4', '2':'ECC10', '3':'eos', '4':'Blood', 
         '5':'RBC_2', '6':'monos', '7':'neutro', '8':'lym_T', '100':'new'})
    em_df['image'] = list(map(embeddable_image, all_frames_reshaped))
    em_df = pd.concat([em_df, all_tables], axis=1)
    
    return[em_df]    

######################################################################################################
######################################### PLOT embedding #############################################
######################################################################################################

def plot_embedding(df_embedding, title, df_embedding_training):

    output_file(title)
    #df_embedding = df_embedding.drop('level_0', axis=1)
    datasource = ColumnDataSource(df_embedding)
    datasource2 = ColumnDataSource(df_embedding_training)

    mapper1 = CategoricalColorMapper(factors=['0', '1', '2', '3', '4' '5', '6', '7', '8'],
                                     palette=Paired[9])
    #mapper = CategoricalColorMapper(factors=['0.0'], palette=Oranges[3])
    mapper2 = LinearColorMapper(palette=Viridis256, low=0, high=0.08) #0-0.1
    color_bar2 = ColorBar(color_mapper=mapper2, location=(0,0), label_standoff=8, major_label_text_font_size="10pt")
    mapper3 = LinearColorMapper(palette=Viridis256, low=20, high=70) #0-100
    color_bar3 = ColorBar(color_mapper=mapper3, location=(0,0), label_standoff=8, major_label_text_font_size="10pt")
    mapper4 = LinearColorMapper(palette=Viridis256, low=26, high=32) #10-40
    color_bar4 = ColorBar(color_mapper=mapper4, location=(0,0), label_standoff=8, major_label_text_font_size="10pt")
    
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

    #button = Button(label = "OK", button_type = "success")
    #select = Select(title="Colour coding:", value="Cell types", options=["Cell types", "Deformation", "Area", "Brightness"])

    plot_figure1 = figure(
            #title=title,
            title='Training data with labels',
            plot_width=600,
            plot_height=400,
            tools=('pan, wheel_zoom, reset'),
            tooltips = tooltips1,
            #x_range=[-15, -7],
            #y_range=[-1, 9]
            x_range=[-22, 15],
            y_range=[-12, 20]
            )
    plot_figure1.circle(
            'x',
            'y',
            source=datasource2,
            color=dict(field='digit', transform=mapper1),   ################### change to 'digit' if we want labels
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4,
            legend = 'digit2'
            )
    plot_figure1.legend.orientation = "vertical"
    plot_figure1.legend.location = "bottom_right"
       
    plot_figure2 = figure(
            #title=title,
            title='Deformation',
            plot_width=600,
            plot_height=400,
            tools=('pan, wheel_zoom, reset'),
            tooltips = tooltips1,
            #x_range=[-15, -7],
            #y_range=[-1, 9]
            x_range=plot_figure1.x_range,
            y_range=plot_figure1.y_range
            )
    plot_figure2.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='deform', transform=mapper2),   ################### change to 'digit' if we want labels
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4
            )
    plot_figure2.add_layout(color_bar2, 'right')
    
    plot_figure3 = figure(
            title='Area',
            plot_width=600,
            plot_height=400,
            tools=('pan, wheel_zoom, reset'),
            tooltips = tooltips1,
            #x_range=[-15, -7],
            #y_range=[-1, 9]
            x_range=plot_figure1.x_range,
            y_range=plot_figure1.y_range
            )
    plot_figure3.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='area_um', transform=mapper3),   ################### change to 'digit' if we want labels
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4
            )
    plot_figure3.add_layout(color_bar3, 'right')

    plot_figure4 = figure(
            title='Brightness',
            plot_width=600,
            plot_height=400,
            tools=('pan, wheel_zoom, reset'),
            tooltips = tooltips1,
            #x_range=[-15, -7],
            #y_range=[-1, 9]
            x_range=plot_figure1.x_range,
            y_range=plot_figure1.y_range
            )
    plot_figure4.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='bright_avg', transform=mapper4),   ################### change to 'digit' if we want labels
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4
            )
    plot_figure4.add_layout(color_bar4, 'right')
    
#    callback_function = CustomJS(args=dict(source=datasource), code="""
#                        var data = source.data;
#                        var f = cb_obj.value
#                    """)
    
    grid = gridplot([[plot_figure1, plot_figure2], [plot_figure3, plot_figure4]])
    show(grid)
    
    #layout = column(widgetbox(select), widgetbox(button), plot_figure)
    #layout = row(plot_figure, plot_figure2)
    #show(layout)
    
    return[]
    
######################################################################################################
################################ PLOT comparison of embeddings #######################################
######################################################################################################

def plot_comparison(df_embedding_new, title, df_embedding_training):

    output_file(title)
    datasource = ColumnDataSource(df_embedding_training)
    datasource2 = ColumnDataSource(df_embedding_new)

    mapper1 = CategoricalColorMapper(factors=['0', '1', '2', '3', '4' '5', '6', '7', '8'],
                                     palette=Paired[9])
    
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

    #button = Button(label = "OK", button_type = "success")
    #select = Select(title="Colour coding:", value="Cell types", options=["Cell types", "Deformation", "Area", "Brightness"])

    plot_figure1 = figure(
            #title=title,
            title='Training data with labels (used for training)',
            plot_width=600,
            plot_height=400,
            tools=('pan, wheel_zoom, reset'),
            tooltips = tooltips1,
            #x_range=[-15, -7],
            #y_range=[-1, 9]
            x_range=[-22, 15],
            y_range=[-12, 20]
            )
    plot_figure1.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='digit', transform=mapper1),   ################### change to 'digit' if we want labels
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4,
            legend = 'digit2'
            )
    plot_figure1.legend.orientation = "vertical"
    plot_figure1.legend.location = "bottom_right"

    plot_figure2 = figure(
            #title=title,
            title='Testing data',
            plot_width=600,
            plot_height=400,
            tools=('pan, wheel_zoom, reset'),
            tooltips = tooltips1,
            x_range=plot_figure1.x_range,
            y_range=plot_figure1.y_range
            )
    plot_figure2.circle(
            'x',
            'y',
            source=datasource2,
            color=dict(field='digit', transform=mapper1),   ################### change to 'digit' if we want labels
            line_alpha=0.6,
            fill_alpha=0.6,
            size=4,
            legend = 'digit2'
            )
    plot_figure2.legend.orientation = "vertical"
    plot_figure2.legend.location = "bottom_right"

    layout = row(plot_figure1, plot_figure2)
    show(layout)
    
    return[]
    
###########################################################################################################
##################################         TRAINING DATASET         #######################################
###########################################################################################################
# LOAD DATA
    
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

###########################################################################################################
# EMBEDDING

# train UMAP using training data and labels    
reducer = umap.UMAP(n_neighbors=15) # min_dist, n_components, metric

#supervised UMAP
print("------------ Embedding in progress (supervised) --------------")
start = time. time()
embedding = reducer.fit(all_frames_b, all_labels)
end = time. time()
print("------------UMAP embedding (supervised) finished, embedding time = ",
      (end - start), "--------------")

#unsupervised UMAP
print("------------ Embedding in progress (unsupervised) --------------")
start = time. time()
embedding2 = reducer.fit(all_frames_b)
end = time. time()
print("------------ UMAP embedding (unsupervised) finished, embedding time = ",
      (end - start), "--------------")

#saves reducer (the model to be used for future embeddings) to disk with todays date:
date_string = datetime.strftime(datetime.date(datetime.now()), '%Y%m%d')
embedding_model_filename = (date_string + "_embedding_BTNK_neigh30_3D.sav")
pickle.dump(embedding, open(embedding_model_filename, 'wb'))

###########################################################################################################
# CREATE DATAFRAME AND PLOT EMBEDDING

#create dataframe of embedding and plot the embedding - SUPERVISED
training_embedding_df = create_df(embedding.embedding_, all_frames, all_labels, all_tables)  ############### change to loaded_embedding if needed
training_embedding_df = training_embedding_df[0]
training_embedding_df = training_embedding_df.drop('level_0', axis=1) ##used to be in plotting funciton at line 4
title_training = "Embedding_of_training_data_supervised.html"
plot_embedding(training_embedding_df, title_training, training_embedding_df)
print("------------Training embedding plotted (supervised) -------------- ")

#create dataframe of embedding and plot the embedding - UNSUPERVISED
training_embedding_df2 = create_df(embedding2.embedding_, all_frames, all_labels, all_tables)
training_embedding_df2 = training_embedding_df2[0]
training_embedding_df2 = training_embedding_df2.drop('level_0', axis=1) ##used to be in plotting funciton at line 4
title_training2 = "Embedding_of_training_data_unsupervised.html"
plot_embedding(training_embedding_df2, title_training2, training_embedding_df2)
print("------------Training embedding plotted (unsupervised)-------------- ")


####################################################################################################################
################################################# TESTING DATASET ##################################################
####################################################################################################################

#load new file, create fake labels
df_new, frames_new, table_new = framecapture(
        r"E:\UMAP_repo\20190208_CD3_CD56_CD19_basos_ex2.")
#        r"E:\UMAP_repo\20180211_3_45_19_unknown.")
#        r"E:\UMAP_repo\20190208_CD3_CD56_CD19_Bcells2.")
#       r"E:\UMAP_repo\20190208_CD3_CD56_CD19_CD3neg_NK.")

labels_new = np.zeros(frames_new.shape[0])+100

##loads reducer (if we want to use saved embedding rather than a newly generated one)
loaded_embedding = pickle.load(open("20190227_embedding_1.sav", 'rb'))

#use trained embedding to embed new points (frames_new)
newembedding = new_embedding(loaded_embedding, frames_new)        ####### here "load_embedding", or use newly trained "embedding"

#create dataframe with embedding, frames, labels (fake 0) and table with parameters
df_newembedding = create_df(newembedding[0], frames_new, labels_new, table_new)

#create plot of new embedding
title = "Basophils2.html"
plot_embedding(df_newembedding[0], title, training_embedding_df)


####################################################################################################################
########################################### split training + testing data ##########################################
####################################################################################################################
#X = array[:,0:8]
#Y = array[:,8]
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
## Fit the model on 33%
#model = LogisticRegression()
#model.fit(X_train, Y_train)


###################################################################################################################
### loading day 2 for testing

df_Bcells2, frames_Bcells2, table_Bcells2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_Bcells2.")
df_CD3neg_NK2, frames_CD3neg_NK2, table_CD3neg_NK2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_CD3neg_NK2.")
df_debris2, frames_debris2, table_debris2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_debris.")
df_eos2, frames_eos2, table_eos2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_eos.")
df_ery2, frames_ery2, table_ery2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_ery2.")
df_erydub2, frames_erydub2, table_erydub2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_erydub.")
df_lymphos_ex2, frames_lymphos_ex2, table_lymphos_ex2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_lymphos.")
df_monos2, frames_monos2, table_monos2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_monos.")
df_neutro2, frames_neutro2, table_neutro2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_neutro.")
df_Tcells2, frames_Tcells2, table_Tcells2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_Tcells2.")
print("------------ Training images loaded--------------")

# concatenate training images to 1 matrix, concatenate corresponding labels to vector with labels
frames2 = [frames_Bcells2, frames_CD3neg_NK2, frames_debris2, frames_eos2, frames_ery2, frames_erydub2, frames_monos2, frames_neutro2, frames_Tcells2]
all_frames2 = pd.concat(frames2)
all_frames_b2 = np.nan_to_num(all_frames2)

#create and concatenate labels
all_labels2= np.concatenate((np.zeros(frames_Bcells2.shape[0]), np.zeros(frames_CD3neg_NK2.shape[0])+1,
                           np.zeros(frames_debris2.shape[0])+2, np.zeros(frames_eos2.shape[0])+3, 
                           np.zeros(frames_ery2.shape[0])+4, np.zeros(frames_erydub2.shape[0])+5,
                           np.zeros(frames_monos2.shape[0])+6, np.zeros(frames_neutro2.shape[0])+7, 
                           np.zeros(frames_Tcells2.shape[0])+8 ))
all_labels2 = all_labels2.astype(int)
all_labels_df2 = pd.DataFrame(all_labels2, columns=['labels'])

#concatenate tables
tables2 = [table_Bcells2, table_CD3neg_NK2, table_debris2, table_eos2, table_ery2, table_erydub2, table_monos2, table_neutro2, table_Tcells2]
all_tables2 = pd.concat(tables2)
all_tables2 = all_tables2.reset_index()

#use trained embedding to embed new points 
newembedding_all = new_embedding(embedding, all_frames_b2)

#create dataframe with embedding, frames, labels (fake 0) and table with parameters
df_newembedding_all = create_df(newembedding_all[0], all_frames2, all_labels2, all_tables2)
df_newembedding_all = df_newembedding_all[0]
df_newembedding_all = df_newembedding_all.drop('level_0', axis=1)

#create plot of new embedding
title = "New_embedding_day_2.html"
plot_comparison(df_newembedding_all, title, training_embedding_df)


###################################################################################################################

##for 3d plotting>
#from mpl_toolkits.mplot3d import Axes3D
#data = np.random.rand(11311, 4) #4 values giving a colour
#fig2 = plt.figure()
#ax2=fig2.add_subplot(111, projection = '3d')
##ax.scatter(embedding.embedding_[:,0], embedding.embedding_[:,1], embedding.embedding_[:,2], c=data, s=100)
##newembedding_all=newembedding_all[0]
#ax2.scatter(newembedding_all[:,0], newembedding_all[:,1], newembedding_all[:,2], c=data, s=100)

###################################################################################################################
################################### spiked blood unsupervised embedding ###########################################

# ECC10, ECC4, WahT spiked blood
df_new3a, frames_new3a, table_new3a = framecapture(r"E:\UMAP_repo\20190222_Marketa_SpikedBlood\20190222_ECC4_Cyto13_blood.")
df_new3b, frames_new3b, table_new3b = framecapture(r"E:\UMAP_repo\20190222_Marketa_SpikedBlood\20190222_ECC10_Cyto13_blood_2.")
df_new3c, frames_new3c, table_new3c = framecapture(r"E:\UMAP_repo\20190222_Marketa_SpikedBlood\20190222_Waht_Cyto13_blood.")

#frames_new3 = [frames_new3a, frames_new3b, frames_new3c]
frames_new3 = [frames_new3a, frames_new3b]
frames_new3 = pd.concat(frames_new3)
frames_new3=np.nan_to_num(frames_new3) # only this stays if only one file

#labels_new3 = np.zeros(frames_new3.shape[0])+100 #if only one file
label_new3a = pd.to_numeric(table_new3a['fl1_max'])
for i in range(0,label_new3a.shape[0]):
    if label_new3a.iloc[i] >= 1000:
        label_new3a.iloc[i] = 1
    else: label_new3a.iloc[i] = 4
label_new3b = pd.to_numeric(table_new3b['fl1_max'])
for i in range(0,label_new3b.shape[0]):
    if label_new3b.iloc[i] >= 1000:
        label_new3b.iloc[i] = 2
    else: label_new3b.iloc[i] = 4
label_new3c = pd.to_numeric(table_new3c['fl1_max'])
for i in range(0,label_new3c.shape[0]):
    if label_new3c.iloc[i] >= 1000:
        label_new3c.iloc[i] = 6
    else: label_new3c.iloc[i] = 4

#labels_new3= pd.concat([label_new3a, label_new3b, label_new3c], ignore_index = True)
labels_new3= pd.concat([label_new3a, label_new3b], ignore_index = True)
#labels_new3= np.concatenate(label_new3a, label_new3b, label_new3c)
labels_new3 = labels_new3.astype(int)
labels_new3_df = pd.DataFrame(labels_new3, columns=['labels'])

#tables_new3 = [table_new3a, table_new3b, table_new3c]
tables_new3 = [table_new3a, table_new3b]
tables_new3 = pd.concat(tables_new3)
tables_new3 = tables_new3.reset_index()

reducer = umap.UMAP(n_neighbors=20) 
#unsupervised UMAP
print("------------ Embedding in progress (unsupervised) --------------")
start = time. time()
embedding2 = reducer.fit(frames_new3)
end = time. time()
print("------------ UMAP embedding (unsupervised) finished, embedding time = ",
      (end - start), "--------------")

print("------------ Embedding in progress (supervised) --------------")
start = time. time()
embedding3 = reducer.fit(frames_new3, labels_new3)
end = time. time()
print("------------UMAP embedding (supervised) finished, embedding time = ",
      (end - start), "--------------")

#create dataframe of embedding and plot the embedding - UNSUPERVISED
frames_new3=pd.DataFrame(frames_new3)
training_embedding_df2 = create_df(embedding2.embedding_, frames_new3, labels_new3, tables_new3)
training_embedding_df2 = training_embedding_df2[0]
training_embedding_df2 = training_embedding_df2.drop('level_0', axis=1) ##used to be in plotting funciton at line 4
title_training2 = "Embedding_of_training_data_unsupervised.html"
plot_embedding(training_embedding_df2, title_training2, training_embedding_df2)
print("------------Training embedding plotted (unsupervised)-------------- ")

#create dataframe of embedding and plot the embedding - SUPERVISED
frames_new3=pd.DataFrame(frames_new3)
training_embedding_df3 = create_df(embedding3.embedding_, frames_new3, labels_new3, tables_new3)
training_embedding_df3 = training_embedding_df3[0]
training_embedding_df3 = training_embedding_df3.drop('level_0', axis=1) ##used to be in plotting funciton at line 4
title_training3 = "Embedding_of_training_data_supervised.html"
plot_embedding(training_embedding_df3, title_training3, training_embedding_df3)
print("------------Training embedding plotted (unsupervised)-------------- ")
