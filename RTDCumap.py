# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




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
blood_image_new = np.squeeze(blood_image_cropped2)                              #remove single dimension

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

#embed and time
import time
start = time. time()
embedding = reducer.fit_transform(blood_image_new_flat, y=blood_labels_numbers2)
#embedding = reducer.fit_transform(blood_image_new_flat)
embedding.shape
end = time. time()
print("------------UMAP embedding finished, embedding time = ------------------")
print(end - start)

#scatter plot of embedding
sns.scatterplot(embedding[:, 0], embedding[:, 1], hue = blood_labels2, marker='o', size=1)


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

bloodembedding_df = pd.DataFrame(embedding, columns=('x', 'y'))
bloodembedding_df['digit'] = [str(x) for x in blood_labels_numbers2]
bloodembedding_df['image'] = list(map(embeddable_image, blood_image_new))


output_file("testbokeh.html")
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

bloodembedding_df_2=bloodembedding_df
