
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans # import k-means from clustering stage
import matplotlib.cm as cm
import matplotlib.colors as colors
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe


# In[2]:


#Getting the csv file with coordinates and storing it in a dataframe
get_ipython().system(u'wget --quiet https://raw.githubusercontent.com/maydinel/Coursera_Capstone/master/map%20-%20kadikoy%20-%20v5.csv')
df_coord = pd.read_csv('map - kadikoy - v5.csv')
print(df_coord.shape)
df_coord.head()


# In[3]:


#Dropping the unnecessary columns
df_coord.drop(['marker-color', 'marker-size', 'marker-symbol'], axis=1, inplace = True)
print(df_coord.shape)
df_coord.head()


# In[4]:


get_ipython().system(u'conda install -c conda-forge folium=0.5.0 --yes # installing folium')
import folium # map rendering library


# In[5]:


#Creating map of Kadikoy
map_kadikoy = folium.Map(location=[40.9890, 29.0275], zoom_start=15)

#Adding markers to the map
for lat, lon, name in zip(df_coord['lat'], df_coord['lon'], df_coord['name']):
    label = '{}'.format(name)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_kadikoy)  
    
map_kadikoy


# Entering the Credentials for the Foursquare API

# In[6]:


# @hidden_cell

CLIENT_ID = 'EAUSZ3BTGKL3XCTW5CFM3DH1YUG5XL4A3R1SHE2FM4YR2HBA' #Foursquare ID
CLIENT_SECRET = 'PMYI2KYXCVXFACD14DH0IDQR2R0GT1FV1XYRJNBKT1RJXMQX' #Foursquare Secret
VERSION = '20190625' # Foursquare API version

print('Credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[9]:


#Checking results for a certain data point
df_coord.loc[0, 'name']


# In[10]:


neighborhood_latitude = df_coord.loc[0, 'lat'] # neighborhood latitude value
neighborhood_longitude = df_coord.loc[0, 'lon'] # neighborhood longitude value

neighborhood_name = df_coord.loc[0, 'name'] # region code

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# In[11]:


# @hidden cell

LIMIT = 10000 # limit of number of venues returned by Foursquare API

radius = 150 # define radius

# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[12]:


#Retrieving the results from the Foursquare Places API
results = requests.get(url).json()
results


# In[13]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[14]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[15]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[16]:


#Function for getting the venues nearby
def getNearbyVenues(name, lat, lng, radius=150):
    
    venues_list=[]
    for name, lat, lng in zip(name, lat, lng):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[17]:


kadikoy_venues = getNearbyVenues(name=df_coord['name'],
                                   lat=df_coord['lat'],
                                   lng=df_coord['lon']
                                  )


# In[23]:


print(kadikoy_venues.shape)
kadikoy_venues.head()


# In[25]:


kadikoy_venues


# In[27]:


kadikoy_venues.groupby('Neighborhood').count()


# In[28]:


print('There are {} unique categories.'.format(len(kadikoy_venues['Venue Category'].unique())))


# In[29]:


# one hot encoding
kadikoy_onehot = pd.get_dummies(kadikoy_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
kadikoy_onehot['Neighborhood'] = kadikoy_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [kadikoy_onehot.columns[-1]] + list(kadikoy_onehot.columns[:-1])
kadikoy_onehot = kadikoy_onehot[fixed_columns]

kadikoy_onehot.head()


# In[30]:


kadikoy_onehot.shape


# In[31]:


kadikoy_grouped = kadikoy_onehot.groupby('Neighborhood').mean().reset_index()
kadikoy_grouped


# In[32]:


kadikoy_grouped.shape


# In[33]:


num_top_venues = 5

for hood in kadikoy_grouped['Neighborhood']:
    print("----"+str(hood)+"----")
    temp = kadikoy_grouped[kadikoy_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[34]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[35]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = kadikoy_grouped['Neighborhood']

for ind in np.arange(kadikoy_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(kadikoy_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[36]:


# set number of clusters
kclusters = 5

kadikoy_grouped_clustering = kadikoy_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(kadikoy_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[37]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

kadikoy_merged = df_coord

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
kadikoy_merged = kadikoy_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='name')

kadikoy_merged.head() # check the last columns!


# In[38]:


# create map
map_clusters = folium.Map(location=[40.9890, 29.0275], zoom_start=15)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(kadikoy_merged['lat'], kadikoy_merged['lon'], kadikoy_merged['name'], kadikoy_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# Observations about the Clusters:

# In[39]:


kadikoy_merged.loc[kadikoy_merged['Cluster Labels'] == 0, kadikoy_merged.columns[[2] + list(range(3, kadikoy_merged.shape[1]))]]


# In[40]:


kadikoy_merged.loc[kadikoy_merged['Cluster Labels'] == 1, kadikoy_merged.columns[[2] + list(range(3, kadikoy_merged.shape[1]))]]


# In[41]:


kadikoy_merged.loc[kadikoy_merged['Cluster Labels'] == 2, kadikoy_merged.columns[[2] + list(range(3, kadikoy_merged.shape[1]))]]


# In[42]:


kadikoy_merged.loc[kadikoy_merged['Cluster Labels'] == 3, kadikoy_merged.columns[[2] + list(range(3, kadikoy_merged.shape[1]))]]


# In[43]:


kadikoy_merged.loc[kadikoy_merged['Cluster Labels'] == 4, kadikoy_merged.columns[[2] + list(range(3, kadikoy_merged.shape[1]))]]

