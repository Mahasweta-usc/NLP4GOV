import ipywidgets as widgets
from ipywidgets import HBox, Label
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider
import time
import pandas as pd

import json

# Opening JSON file
f = open('/content/NLP4GOV/src/policy_comparison/data/reddit_rules_top_100.json')

# returns JSON object as
# a dictionary
data = json.load(f)
subs_list = list(data.keys())

#Create DF
df = pd.DataFrame(columns = ['Dropdown_column', 'Float_column'])
df

# Layout
form_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
)


button_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='center',
    padding = '5%'
)


# Independent dropdown item

drop_down_input = 'Dropdown_input_1'
drop_down = widgets.Dropdown(options=subs_list)


# Dependent drop down
# Dependent drop down elements
dependent_drop_down_input = 'Dropdown_input_2'
dependent_drop_down = widgets.Dropdown(options=subs_list)

# Define dependent drop down

# dependent_drop_down = widgets.Dropdown(options=(dependent_drop_down_elements))

def dropdown_handler(change):
    global drop_down_input
    # print('\r','Dropdown: ' + str(change.new),end='')
    drop_down_input = change.new
drop_down.observe(dropdown_handler, names='value')

def dep_dropdown_handler(change):
    global dependent_drop_down_input
    # print('\r','Dropdown: ' + str(change.new),end='')
    dependent_drop_down_input = change.new
dependent_drop_down.observe(dep_dropdown_handler, names='value')

# Button

button = widgets.Button(description='Add row to dataframe')
out = widgets.Output()
def on_button_clicked(b):
    global df
    button.description = 'Row added'
    time.sleep(1)
    with out:
      new_row = {'Dropdown_column': drop_down_input, 'Float_column': float_input}
      df = df.append(new_row, ignore_index=True)
      button.description = 'Add row to dataframe'
      out.clear_output()
      display(df)
button.on_click(on_button_clicked)

# Form items

form_items = [
    Box([Label(value='Policy Database 1'),
         drop_down], layout=form_item_layout),
    Box([Label(value='Policy Database 2'),
         dependent_drop_down], layout=form_item_layout)
         ]

form = Box(form_items, layout=Layout(
    display='flex',
    flex_flow='column',
    border='solid 1px',
    align_items='stretch',
    width='30%',
    padding = '1%'
))
display(form)
display(out)