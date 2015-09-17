import pandas as pd
import numpy as np

df1 = pd.DataFrame( { 
	"Name" : ["Alice", "Bob", "Mallory", "Mallory", "Bob" , "Mallory"] , 
	"City" : ["Seattle", "Seattle", "Portland",\
	"Seattle", "Seattle", "Portland"] } )

g1 = df1.groupby( "Name" ).count()
g2 = g1.add_suffix('_Count').reset_index()


