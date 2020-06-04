import tensorflow as tf 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
class_value = st.sidebar.selectbox("Class",[0,1,2,3,4,5,6,7,8,9])
random_seed = st.sidebar.number_input("Random seed", value=1)
nums = st.sidebar.number_input("Number of examples", value=20)


@st.cache 
def class_examples(class_value, nums=20, fix_seed=1): 
    if fix_seed > 0: 
        np.random.seed(1)
    
    class_indx = np.where(y_train == class_value)
    examples = x_train[class_indx]
    samples = examples[np.random.choice(examples.shape[0], nums, replace=False), :,:]
    return samples

display_samples = class_examples(class_value, nums, fix_seed=random_seed)

st.sidebar.button("Reload",class_examples(class_value, nums, fix_seed=random_seed))

num_cols = 4.0 
mod_value = display_samples.shape[0] % num_cols 
num_rows  = (display_samples.shape[0] - mod_value)/num_cols


fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(int(num_rows), int(num_cols)),  
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, display_samples):
    # Iterating over the grid returns the Axes.
    
    ax.imshow(im)
st.pyplot()




from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
    row=1, col=2
)


fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")

st.plotly_chart(fig)
