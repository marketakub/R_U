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
        
#        cellimg_cropped_flattened_pd=pd.DataFrame(cellimg_cropped_flattened)
#        print(cellimg_cropped_flattened_pd)
#        
#        if pd.isnull(cellimg_cropped_flattened_pd).any:
#            i = cellimg_cropped_flattened_pd
#            print("List is empty") 
#        else:
#            #print("OK")
#            Images.append(cellimg_cropped_flattened_pd) 
        
        #Images.append(cellimg_cropped)
        #Images.append(cellimg)
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
    #Images.append(df)
    plt.imshow(cellimg_cropped)
    #plt.scatter(pos_x,pos_y,c='w')
    Images=pd.DataFrame(Images)
    
    return[Images, nr_of_images, IMS]
    #return[Images, df]

#plt.figure()
#plt.imshow(frames_unknown[2])
#plt.figure()
#plt.imshow(frames_unknown[160])
    


frames_unknown, labels_unknown, IMS_unknown = framecapture(r"E:\UMAP_repo\20180211_3_45_19_unknown.")
frames_basos_ex, labels_basos_ex, IMS_basos_ex = framecapture(r"E:\UMAP_repo\20180211_3_56_19_basos_ex.")
frames_Bcells, labels_Bcells, IMS_Bcells = framecapture(r"E:\UMAP_repo\20180211_3_56_19_Bcells.")
frames_CD3neg_NK, labels_CD3neg_NK, IMS_CD3neg_NK = framecapture(r"E:\UMAP_repo\20180211_3_56_19_CD3neg_NK.")
frames_CD3pos_NK, labels_CD3pos_NK, IMS_CD3pos_NK = framecapture(r"E:\UMAP_repo\20180211_3_56_19_CD3pos_NK.")
frames_debris, labels_debris, IMS_debris = framecapture(r"E:\UMAP_repo\20180211_3_56_19_debris.")
frames_eos, labels_eos, IMS_eos = framecapture(r"E:\UMAP_repo\20180211_3_56_19_eos.")
frames_ery, labels_ery, IMS_ery = framecapture(r"E:\UMAP_repo\20180211_3_56_19_ery.")
frames_erydub, labels_erydub, IMS_erydub = framecapture(r"E:\UMAP_repo\20180211_3_56_19_erydub.")
frames_lymphos_ex, labels_lymphos_ex, IMS_lymphos_ex = framecapture(r"E:\UMAP_repo\20180211_3_56_19_lymphos_ex.")
frames_monos, labels_monos, IMS_monos = framecapture(r"E:\UMAP_repo\20180211_3_56_19_mono.")
frames_neutro, labels_neutro, IMS_neutro = framecapture(r"E:\UMAP_repo\20180211_3_56_19_neutro.")
frames_Tcells, labels_Tcells, IMS_Tcells = framecapture(r"E:\UMAP_repo\20180211_3_56_19_Tcells.")

frames_unknown2, labels_unknown2, IMS_unknown2 = framecapture(r"E:\UMAP_repo\20180211_14_45_sig8_unknown2.")



frames = [frames_Bcells,frames_CD3neg_NK, frames_debris, frames_eos, frames_ery, frames_erydub, frames_monos, frames_neutro, frames_Tcells]
all_frames = pd.concat(frames)
all_labels=np.concatenate((np.zeros(labels_Bcells), np.zeros(labels_CD3neg_NK)+1,np.zeros(labels_debris)+2, np.zeros(labels_eos)+3, np.zeros(labels_ery)+4, np.zeros(labels_erydub)+5, np.zeros(labels_monos)+6,np.zeros(labels_neutro)+7, np.zeros(labels_Tcells)+8
))
frames_b= np.nan_to_num(all_frames)


#images = [IMS_Bcells, IMS_CD3neg_NK, IMS_debris, IMS_eos, IMS_ery, IMS_erydub, IMS_monos, IMS_neutro, IMS_Tcells]
#images = pd.DataFrame(images)
#all_images = pd.concat(images)


frames_b_test = np.nan_to_num(frames_unknown)
frames_b_test2 = np.nan_to_num(frames_unknown2)
frames_b_CD3pos_NK = np.nan_to_num(frames_CD3pos_NK)
all_labels_test = (np.zeros(labels_unknown))
all_labels_test2 = (np.zeros(labels_unknown2))
all_labels_CD3pos_NK = (np.zeros(labels_CD3pos_NK))


#UMAP embedding
import umap
reducer = umap.UMAP(n_neighbors=15)
print("------------UMAP imported------------------")



##############################################################################################
################################## first embedding #############################################

#embed and time
import time
start = time. time()
embedding = reducer.fit(frames_b, all_labels)
#embedding = reducer.fit(all_frames, all_labels)
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
    image = Image.fromarray(img_data, mode='L').resize((102,72))
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Set1
from bokeh.palettes import Paired

reshaped_all=all_frames.values.reshape(50076,24,34)
reshaped_all=reshaped_all*255+60
plt.imshow(reshaped_all[0,:,:])

bloodembedding_df = pd.DataFrame(embedding.embedding_, columns=('x', 'y'))
#bloodembedding_df['digit'] = all_labels
bloodembedding_df['digit'] = [str(x) for x in all_labels]
bloodembedding_df['image'] = list(map(embeddable_image, reshaped_all))

print(bloodembedding_df['digit'].dtype)

output_file("learned_embedding.html")
datasource = ColumnDataSource(bloodembedding_df)
color_mapping = CategoricalColorMapper(factors=['0.0', '1.0', '2.0', '3.0', '4.0' '5.0', '6.0', '7.0', '8.0'],
                                       palette=Paired[9])

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
    tooltips = tooltips1,
    x_range=[-15, -7],
    y_range=[-1, 9]
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


##############################################################################################
################################## new embedding #############################################

#embed and time
import time
start = time. time()
test_embedding = embedding.transform(frames_b_test)
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)


##############################################################################################

from bokeh.palettes import Oranges

reshaped_unknown=frames_unknown.values.reshape(1150,24,34)
reshaped_unknown=reshaped_unknown*255+60
plt.imshow(reshaped_unknown[0,:,:])


bloodembedding_df_test = pd.DataFrame(test_embedding, columns=('x', 'y'))
bloodembedding_df_test['digit'] = [str(x) for x in all_labels_test]
bloodembedding_df_test['image'] = list(map(embeddable_image, reshaped_unknown))


#print(IMS_unknown("Size"))
#Imgs = []
#j = np.zeros(IMS_unknown[0].shape)
#for x in range (0,10):
#    j= IMS_unknown[x]
#    #i = IMS_unknown[0]
#    Imgs.append(j)
#
#i = IMS_unknown[0]
#plt.imshow(i)


#testimage = np.array(IMS_unknown)
#img_data = testimage.astype(np.uint8)
#image = Image.fromarray(img_data, mode='L')

output_file("newly_embedded_points.html")
datasource2 = ColumnDataSource(bloodembedding_df_test)
color_mapping2 = CategoricalColorMapper(factors=['0.0'],
                                       palette=Oranges[3])
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
plot_figure2 = figure(
    title='UMAP projection of the dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset'),
    tooltips = tooltips1,
    x_range=[-22, 15],
    y_range=[-12, 20]
    )

plot_figure2.circle(
    'x',
    'y',
    source=datasource2,
    color=dict(field='digit', transform=color_mapping2),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
    )

show(plot_figure2)


###############################################################################################
##############################################################################################
################################## new embedding - unkown 2 #############################################

#embed and time
import time
start = time. time()
test_embedding2 = embedding.transform(frames_b_test2)
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)


##############################################################################################

from bokeh.palettes import Oranges

reshaped_unknown2=frames_unknown2.values.reshape(668,24,34)
reshaped_unknown2=reshaped_unknown2*255+60
plt.imshow(reshaped_unknown2[0,:,:])


bloodembedding_df_test = pd.DataFrame(test_embedding2, columns=('x', 'y'))
bloodembedding_df_test['digit'] = [str(x) for x in all_labels_test2]
bloodembedding_df_test['image'] = list(map(embeddable_image, reshaped_unknown2))


output_file("Martins_unknown_2.html")
datasource2 = ColumnDataSource(bloodembedding_df_test)
color_mapping2 = CategoricalColorMapper(factors=['0.0'],
                                       palette=Oranges[3])
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
plot_figure2 = figure(
    title='UMAP projection of the dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset'),
    tooltips = tooltips1,
    x_range=[-22, 15],
    y_range=[-12, 20]
    )

plot_figure2.circle(
    'x',
    'y',
    source=datasource2,
    color=dict(field='digit', transform=color_mapping2),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
    )

show(plot_figure2)

###############################################################################################
###############################################################################################
################################## new embedding - CD3+ NK #############################################

#embed and time
import time
start = time. time()
test_embedding3 = embedding.transform(frames_b_CD3pos_NK)
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)


##############################################################################################

from bokeh.palettes import Oranges

reshaped_CD3pos_NK=frames_CD3pos_NK.values.reshape(747,24,34)
reshaped_CD3pos_NK=reshaped_CD3pos_NK*255+60
plt.imshow(reshaped_CD3pos_NK[0,:,:])


bloodembedding_df_test = pd.DataFrame(test_embedding3, columns=('x', 'y'))
bloodembedding_df_test['digit'] = [str(x) for x in all_labels_CD3pos_NK]
bloodembedding_df_test['image'] = list(map(embeddable_image, reshaped_CD3pos_NK))

output_file("CD3pos_NK.html")
datasource2 = ColumnDataSource(bloodembedding_df_test)
color_mapping2 = CategoricalColorMapper(factors=['0.0'],
                                       palette=Oranges[3])
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
plot_figure3 = figure(
    title='UMAP projection of the dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset'),
    tooltips = tooltips1,
    x_range=[-22, 15],
    y_range=[-12, 20]
    )

plot_figure3.circle(
    'x',
    'y',
    source=datasource2,
    color=dict(field='digit', transform=color_mapping2),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
    )

show(plot_figure3)



###############################################################################################
###############################################################################################
################################## new embedding - T-cells #############################################

#load
frames_Tcells2, labels_Tcells2, IMS_Tcells2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_Tcells2.")
frames_b_Tcells2 = np.nan_to_num(frames_Tcells2)
all_labels_Tcells2 = (np.zeros(labels_Tcells2))
#all_labels_Tcells2.shape[0]
reshaped_Tcells2=frames_Tcells2.values.reshape(all_labels_Tcells2.shape[0],24,34)
reshaped_Tcells2=reshaped_Tcells2*255+60
plt.imshow(reshaped_Tcells2[0,:,:])


#embed and time
import time
start = time. time()
test_embedding4 = embedding.transform(frames_b_Tcells2)
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)


##############################################################################################

from bokeh.palettes import Oranges

bloodembedding_df_test4 = pd.DataFrame(test_embedding4, columns=('x', 'y'))
bloodembedding_df_test4['digit'] = [str(x) for x in all_labels_Tcells2]
bloodembedding_df_test4['image'] = list(map(embeddable_image, reshaped_Tcells2))

output_file("Tcells2.html")
datasource4 = ColumnDataSource(bloodembedding_df_test4)
color_mapping2 = CategoricalColorMapper(factors=['0.0'],
                                       palette=Oranges[3])
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
plot_figure4 = figure(
    title='UMAP projection of the dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset'),
    tooltips = tooltips1,
    x_range=[-15, -7],
    y_range=[-1, 9]
    )

plot_figure4.circle(
    'x',
    'y',
    source=datasource4,
    color=dict(field='digit', transform=color_mapping2),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
    )

show(plot_figure4)

###############################################################################################
###############################################################################################
################################## new embedding - B-cells #############################################

#load
frames_Bcells2, labels_Bcells2, IMS_Bcells2 = framecapture(r"E:\UMAP_repo\20190208_CD3_CD56_CD19_Bcells2.")
frames_b_Bcells2 = np.nan_to_num(frames_Bcells2)
all_labels_Bcells2 = (np.zeros(labels_Bcells2))
#all_labels_Tcells2.shape[0]
reshaped_Bcells2=frames_Bcells2.values.reshape(all_labels_Bcells2.shape[0],24,34)
reshaped_Bcells2=reshaped_Bcells2*255+60
plt.imshow(reshaped_Bcells2[0,:,:])


#embed and time
import time
start = time. time()
test_embedding5 = embedding.transform(frames_b_Bcells2)
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)


##############################################################################################

from bokeh.palettes import Oranges

bloodembedding_df_test5 = pd.DataFrame(test_embedding5, columns=('x', 'y'))
bloodembedding_df_test5['digit'] = [str(x) for x in all_labels_Bcells2]
bloodembedding_df_test5['image'] = list(map(embeddable_image, reshaped_Bcells2))

output_file("Bcells2.html")
datasource5 = ColumnDataSource(bloodembedding_df_test5)
color_mapping2 = CategoricalColorMapper(factors=['0.0'],
                                       palette=Oranges[3])
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
plot_figure5 = figure(
    title='UMAP projection of the dataset',
    plot_width=600,
    plot_height=600,
    tools=('pan, wheel_zoom, reset'),
    tooltips = tooltips1,
    x_range=[-15, -7],
    y_range=[-1, 9]
    )

plot_figure5.circle(
    'x',
    'y',
    source=datasource5,
    color=dict(field='digit', transform=color_mapping2),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
    )

show(plot_figure5)

#
###############################################################################################
################################### new embedding - 1 point #############################################
#
#test=blood_image_new_flat2[0:1,:]
#test_embedding2 = np.zeros((1,2))
##embed and time
#import time
#start = time. time()
#test_embedding2 = embedding.transform(test)
#end = time. time()
#print("------------UMAP embedding finished, embedding time = ------------------")
#print(end - start)
